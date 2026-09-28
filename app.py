from flask import Flask, request, render_template, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import pickle, sqlite3, os, json, math, hashlib
from datetime import datetime
from io import BytesIO
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'bank_loan_prediction.db')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'bank-loan-prediction-dev-secret-change-me')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

try:
    train_df = pd.read_csv(TRAIN_PATH)
except Exception:
    train_df = pd.DataFrame()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        loan_type TEXT NOT NULL,
        applicant_name TEXT,
        inputs_json TEXT NOT NULL,
        result_json TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS saved_applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        application_id INTEGER NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(application_id) REFERENCES applications(id)
    );
    ''')
    conn.commit(); conn.close()


init_db()


def current_user():
    if not session.get('user_id'):
        return None
    conn = db(); user = conn.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone(); conn.close()
    return user


def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please login to access your Bank Loan Prediction workspace.', 'warning')
            return redirect(url_for('login', next=request.path))
        return fn(*args, **kwargs)
    return wrapper


def calculate_financials(income, co_income, loan_amount, term_months, annual_rate=10.5):
    monthly_income = max((income + co_income) / 12.0, 1)
    r = annual_rate / 1200.0
    emi = loan_amount * 1000 * r * ((1+r)**term_months) / (((1+r)**term_months)-1) if r else loan_amount*1000/term_months
    total = emi * term_months
    interest = total - loan_amount*1000
    dti = emi / monthly_income * 100
    affordability = max(0, min(100, 100 - dti))
    return {'emi': round(emi,2), 'total_interest': round(interest,2), 'total_repayment': round(total,2), 'dti': round(dti,1), 'affordability': round(affordability,1)}


def risk_band(approval):
    if approval >= 75: return 'Low Risk'
    if approval >= 50: return 'Moderate Risk'
    return 'High Risk'


def build_features(form):
    gender=form.get('gender','Male'); married=form.get('married','No'); dep=form.get('dependents','0'); education=form.get('education','Graduate'); employed=form.get('employed','No'); area=form.get('area','Urban')
    credit=float(form.get('credit',1)); income=float(form.get('ApplicantIncome',0)); co=float(form.get('CoapplicantIncome',0)); loan=float(form.get('LoanAmount',0)); term=float(form.get('Loan_Amount_Term',360))
    return {
        'gender':gender,'married':married,'dependents':dep,'education':education,'employed':employed,'credit':credit,'area':area,
        'ApplicantIncome':income,'CoapplicantIncome':co,'LoanAmount':loan,'Loan_Amount_Term':term,
        'Total_income':income+co
    }, [[loan,term,credit,income+co,1 if gender=='Male' else 0,1 if married=='Yes' else 0,1 if dep=='1' else 0,1 if dep=='2' else 0,1 if dep=='3+' else 0,1 if education=='Not Graduate' else 0,1 if employed=='Yes' else 0,1 if area=='Semiurban' else 0,1 if area=='Urban' else 0]]


def explain(inputs, approval):
    positives=[]; risks=[]
    if inputs['credit'] >= 1: positives.append('Good credit history')
    else: risks.append('Limited or poor credit history')
    monthly_income=(inputs['Total_income']/12) if inputs['Total_income'] else 0
    ratio=(inputs['LoanAmount']*1000)/(inputs['Total_income'] or 1)
    if ratio < 5: positives.append('Loan amount is moderate relative to income')
    else: risks.append('Requested loan is high relative to annual income')
    if inputs['employed']=='Yes': positives.append('Applicant reports self-employment')
    if inputs['education']=='Graduate': positives.append('Graduate education profile')
    if inputs['Loan_Amount_Term'] >= 360: positives.append('Longer tenure can reduce monthly EMI')
    if inputs['dependents']=='3+': risks.append('Higher dependent count may increase repayment pressure')
    if not positives: positives.append('Model features were processed successfully')
    if not risks: risks.append('No major rule-based risk flag was triggered')
    tips=[]
    if inputs['credit'] < 1: tips.append('Maintain a strong repayment history and review credit records.')
    if ratio >= 5: tips.append('Consider a lower loan amount or a longer tenure to reduce repayment pressure.')
    if inputs['CoapplicantIncome'] == 0: tips.append('A verified co-applicant income may improve affordability assessment where applicable.')
    if not tips: tips.append('Keep income documentation and repayment history consistent.')
    return positives, risks, tips


def feature_importance():
    names=['Loan Amount','Loan Term','Credit History','Total Income','Male','Married','Dependents 1','Dependents 2','Dependents 3+','Not Graduate','Self Employed','Semiurban','Urban']
    vals=getattr(model,'feature_importances_', np.ones(13)/13)
    pairs=sorted(zip(names, [float(v) for v in vals]), key=lambda x:x[1], reverse=True)[:8]
    return [{'name':n,'value':round(v*100,2)} for n,v in pairs]


def heuristic_estimates(inputs, financials, approval):
    income=inputs['Total_income']; loan=inputs['LoanAmount']; term=inputs['Loan_Amount_Term']
    recommended=max(0, min(loan, income*0.45))
    if financials['dti'] <= 35: interest=9.25
    elif financials['dti'] <= 50: interest=11.25
    else: interest=13.5
    credit_score=int(max(520,min(820, 520 + approval*2.5 + (40 if inputs['credit']>=1 else 0))))
    return {'recommended_loan':round(recommended,1),'estimated_rate':interest,'estimated_credit_score':credit_score,'default_risk':round(100-approval,1),'repayment_capacity':round(financials['affordability'],1)}


def make_prediction(form):
    inputs, features = build_features(form)
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    classes = [str(x) for x in model.classes_]
    pmap = {c: float(p) for c,p in zip(classes, probs)}
    approval = pmap.get('Y',0)*100
    rejection = pmap.get('N',0)*100
    financials=calculate_financials(inputs['ApplicantIncome'],inputs['CoapplicantIncome'],inputs['LoanAmount'],int(inputs['Loan_Amount_Term']))
    band=risk_band(approval)
    positives,risks,tips=explain(inputs,approval)
    extra=heuristic_estimates(inputs,financials,approval)
    # Lightweight anomaly screening for portfolio/demo use (not a fraud verdict).
    anomaly_flag = 'Review recommended' if inputs['LoanAmount'] > max(inputs['Total_income']*0.6, 250) or financials['dti'] > 55 else 'No obvious anomaly'
    result={'prediction':'Approved' if prediction=='Y' else 'Rejected','approval':round(approval,1),'rejection':round(rejection,1),'risk_band':band,'financials':financials,'extra':extra,'anomaly_flag':anomaly_flag,'positives':positives,'risks':risks,'tips':tips,'features':feature_importance()}
    return inputs,result


@app.context_processor
def inject_globals():
    return {'user': current_user(), 'year': datetime.now().year}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        name=request.form.get('name','').strip(); email=request.form.get('email','').strip().lower(); password=request.form.get('password','')
        if not name or not email or len(password)<6:
            flash('Enter a name, valid email and password of at least 6 characters.','danger'); return render_template('register.html')
        try:
            conn=db(); conn.execute('INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)',(name,email,generate_password_hash(password),datetime.now().isoformat(timespec='seconds'))); conn.commit(); conn.close()
            flash('Account created. Please login.','success'); return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An account with this email already exists.','warning')
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        email=request.form.get('email','').strip().lower(); password=request.form.get('password','')
        conn=db(); u=conn.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone(); conn.close()
        if u and check_password_hash(u['password_hash'],password):
            session['user_id']=u['id']; flash('Welcome back to Bank Loan Prediction AI.','success'); return redirect(request.args.get('next') or url_for('dashboard'))
        flash('Invalid email or password.','danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); flash('You have been logged out.','info'); return redirect(url_for('home'))


@app.route('/profile', methods=['GET','POST'])
@login_required
def profile():
    if request.method=='POST':
        name=request.form.get('name','').strip()
        if name:
            conn=db(); conn.execute('UPDATE users SET name=? WHERE id=?',(name,session['user_id'])); conn.commit(); conn.close(); flash('Profile updated.','success')
        return redirect(url_for('profile'))
    return render_template('profile.html')

@app.route('/dashboard')
@login_required
def dashboard():
    conn=db(); rows=conn.execute('SELECT * FROM applications WHERE user_id=? ORDER BY id DESC',(session['user_id'],)).fetchall(); conn.close()
    approvals=[]
    for r in rows: approvals.append(json.loads(r['result_json']).get('approval',0))
    stats={'applications':len(rows),'avg_approval':round(sum(approvals)/len(approvals),1) if approvals else 0,'high_risk':sum(1 for r in rows if json.loads(r['result_json']).get('risk_band')=='High Risk')}
    return render_template('dashboard.html',applications=rows[:6],stats=stats)

@app.route('/apply', methods=['GET','POST'])
@login_required
def apply():
    if request.method=='POST':
        try:
            inputs,result=make_prediction(request.form)
            loan_type=request.form.get('loan_type','Personal Loan'); applicant_name=request.form.get('applicant_name') or user_name()
            conn=db(); cur=conn.execute('INSERT INTO applications(user_id,created_at,loan_type,applicant_name,inputs_json,result_json) VALUES(?,?,?,?,?,?)',(session['user_id'],datetime.now().isoformat(timespec='seconds'),loan_type,applicant_name,json.dumps(inputs),json.dumps(result))); app_id=cur.lastrowid; conn.commit(); conn.close()
            return redirect(url_for('result',application_id=app_id))
        except Exception as e:
            flash(f'Prediction error: {e}','danger')
    return render_template('application.html')

def user_name():
    u=current_user(); return u['name'] if u else 'Applicant'

@app.route('/result/<int:application_id>')
@login_required
def result(application_id):
    conn=db(); r=conn.execute('SELECT * FROM applications WHERE id=? AND user_id=?',(application_id,session['user_id'])).fetchone(); conn.close()
    if not r: return redirect(url_for('dashboard'))
    return render_template('result.html',application=r,result=json.loads(r['result_json']),inputs=json.loads(r['inputs_json']))

@app.route('/history')
@login_required
def history():
    conn=db(); rows=conn.execute('SELECT * FROM applications WHERE user_id=? ORDER BY id DESC',(session['user_id'],)).fetchall(); conn.close(); return render_template('history.html',applications=rows)

@app.route('/save/<int:application_id>', methods=['POST'])
@login_required
def save_application(application_id):
    conn=db(); conn.execute('INSERT INTO saved_applications(user_id,application_id) SELECT ?,? WHERE NOT EXISTS (SELECT 1 FROM saved_applications WHERE user_id=? AND application_id=?)',(session['user_id'],application_id,session['user_id'],application_id)); conn.commit(); conn.close(); flash('Application saved.','success'); return redirect(request.referrer or url_for('history'))

@app.route('/analytics')
@login_required
def analytics():
    conn=db(); rows=conn.execute('SELECT * FROM applications WHERE user_id=? ORDER BY id ASC',(session['user_id'],)).fetchall(); conn.close()
    data=[]
    for r in rows: data.append(json.loads(r['result_json']))
    return render_template('analytics.html',applications=rows,data=data,features=feature_importance())

@app.route('/calculators')
@login_required
def calculators(): return render_template('calculators.html')

@app.route('/assistant', methods=['GET','POST'])
@login_required
def assistant():
    answer=None
    if request.method=='POST':
        q=request.form.get('message','').lower()
        if 'emi' in q: answer='EMI depends on loan amount, annual interest rate and tenure. Use the EMI calculator in the Financial Tools section.'
        elif 'risk' in q: answer='Your risk band is based on the model-estimated approval probability plus transparent financial checks. It is not a bank credit decision.'
        elif 'improve' in q: answer='Common improvement areas are credit history, reducing the requested loan relative to income, and keeping repayment pressure manageable.'
        elif 'dti' in q or 'debt' in q: answer='Debt-to-income ratio compares monthly debt repayment against monthly income. Lower values generally indicate more repayment headroom.'
        else: answer='I can help with EMI, affordability, DTI, loan amount, risk factors, and ways to improve an application.'
    return render_template('assistant.html',answer=answer)

@app.route('/report/<int:application_id>')
@login_required
def report(application_id):
    conn=db(); r=conn.execute('SELECT * FROM applications WHERE id=? AND user_id=?',(application_id,session['user_id'])).fetchone(); conn.close()
    if not r: return redirect(url_for('dashboard'))
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        result=json.loads(r['result_json']); inputs=json.loads(r['inputs_json']); buf=BytesIO(); doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36); styles=getSampleStyleSheet(); story=[Paragraph('Bank Loan Prediction AI — Loan Risk Assessment Report',styles['Title']),Spacer(1,14)]
        story += [Paragraph(f"Applicant: {r['applicant_name']}",styles['Normal']),Paragraph(f"Loan Type: {r['loan_type']}",styles['Normal']),Spacer(1,12)]
        table=Table([['Metric','Value'],['Prediction',result['prediction']],['Approval Probability',f"{result['approval']}%"],['Rejection Probability',f"{result['rejection']}%"],['Risk Band',result['risk_band']],['Model Risk Score',f"{result['extra']['default_risk']}/100"],['Recommended Loan',f"{result['extra']['recommended_loan']} (dataset units)"]],colWidths=[220,240]); table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#172033')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),0.5,colors.grey),('PADDING',(0,0),(-1,-1),8)])); story += [table,Spacer(1,14),Paragraph('Positive Factors',styles['Heading2'])]+[Paragraph('• '+x,styles['Normal']) for x in result['positives']]+[Paragraph('Risk Factors',styles['Heading2'])]+[Paragraph('• '+x,styles['Normal']) for x in result['risks']]+[Paragraph('Recommendations',styles['Heading2'])]+[Paragraph('• '+x,styles['Normal']) for x in result['tips']]
        doc.build(story); buf.seek(0); return send_file(buf,as_attachment=True,download_name=f'Bank Loan Prediction_Report_{application_id}.pdf',mimetype='application/pdf')
    except ImportError:
        flash('PDF export needs reportlab. Run pip install -r requirements.txt.','warning'); return redirect(url_for('result',application_id=application_id))

@app.route('/api/quick-calculation',methods=['POST'])
def quick_calculation():
    data=request.get_json(force=True); f=calculate_financials(float(data.get('income',0)),float(data.get('co_income',0)),float(data.get('loan',0)),int(data.get('term',360)),float(data.get('rate',10.5))); return f

@app.route('/about')
def about(): return render_template('about.html')
@app.route('/contact')
def contact(): return render_template('contact.html')
@app.route('/predict')
def predict(): return redirect(url_for('apply'))

if __name__=='__main__':
    app.run(debug=True)
