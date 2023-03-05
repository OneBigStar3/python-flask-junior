from main import (
    app, mysql, MySQLdb, render_template,
    request, redirect, url_for, session, loggedin,
    hashlib, os, dt_format)

import datetime

# http://localhost:5000/pythonlogin/admin/ - admin home page, view all accounts
@app.route('/pythonlogin/admin/', methods=['GET', 'POST'])
def admin():
    # Check if admin is logged-in
    print('admin url is called')
    if not admin_loggedin():
        return redirect(url_for('logout'))
    msg = ''
    # Retrieve all accounts from the database
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM accounts')
    accounts = cursor.fetchall()
    return render_template('admin/index.html', accounts=accounts)

# http://localhost:5000/pythonlogin/admin/account - create or edit account
@app.route('/pythonlogin/admin/account/<int:id>', methods=['GET', 'POST'])
@app.route('/pythonlogin/admin/account', methods=['GET', 'POST'], defaults={'id': None})
def admin_account(id):
    # Check if admin is logged-in
    if not admin_loggedin():
        return redirect(url_for('logout'))
    page = 'Create'
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # Default input account values
    account = {
        'username': '',
        'password': '',
        'email': '',
        'activation_code': '',
        'rememberme': '',
        'role': 'Member'
    }
    roles = ['Member', 'Admin'];
    # GET request ID exists, edit account
    if id:
        # Edit an existing account
        page = 'Edit'
        # Retrieve account by ID with the GET request ID
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (id,))
        account = cursor.fetchone()
        if request.method == 'POST' and 'submit' in request.form:
            # update account
            password = account['password']
            if account['password'] != request.form['password']:
                 hash = request.form['password'] + app.secret_key
                 hash = hashlib.sha1(hash.encode())
                 password = hash.hexdigest();
            cursor.execute('UPDATE accounts SET username = %s, password = %s, email = %s, activation_code = %s, rememberme = %s, role = %s WHERE id = %s', (request.form['username'],password,request.form['email'],request.form['activation_code'],request.form['rememberme'],request.form['role'],id,))
            mysql.connection.commit()
            return redirect(url_for('admin'))
        if request.method == 'POST' and 'delete' in request.form:
            # delete account
            cursor.execute('DELETE FROM accounts WHERE id = %s', (id,))
            mysql.connection.commit()
            return redirect(url_for('admin'))
    if request.method == 'POST' and request.form['submit']:
        # Create new account
        hash = request.form['password'] + app.secret_key
        hash = hashlib.sha1(hash.encode())
        password = hash.hexdigest();
        cursor.execute('INSERT INTO accounts (username,password,email,activation_code,rememberme,role) VALUES (%s,%s,%s,%s,%s,%s)', (request.form['username'],password,request.form['email'],request.form['activation_code'],request.form['rememberme'],request.form['role'],))
        mysql.connection.commit()
        return redirect(url_for('admin'))
    return render_template('admin/account.html', account=account, page=page, roles=roles)

# http://localhost:5000/pythonlogin/admin/emailtemplate - admin email template page, edit the existing template
@app.route('/pythonlogin/admin/emailtemplate', methods=['GET', 'POST'])
def admin_emailtemplate():
    # Check if admin is logged-in
    if not admin_loggedin():
        return redirect(url_for('logout'))
    # Get the template directory path
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    # Update the template file on save
    if request.method == 'POST':
        content = request.form['content'].replace('\r', '')
        open(template_dir + '/activation-email-template.html', mode='w', encoding='utf-8').write(content)
    # Read the activation email template
    content = open(template_dir + '/activation-email-template.html', mode='r', encoding='utf-8').read()
    return render_template('admin/email-template.html', content=content)
    
# http://localhost:5000/pythonlogin/admin/loginreports -report template page, edit the existing template
@app.route('/pythonlogin/admin/loginreports', methods=['GET', 'POST'])
def login_reports():
    # Check if admin is logged-in
    if not admin_loggedin():
        return redirect(url_for('logout'))
    # Get the template directory path
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
  # Update the template file on save
    if request.method == 'POST':
        content = request.form['content'].replace('\r', '')
        open(template_dir + '/loginreports.html', mode='w', encoding='utf-8').write(content)
    # Read the activation email template
    content = open(template_dir + '/loginreports.html', mode='r', encoding='utf-8').read()
    return render_template('admin/loginreports.html', content=content)
    
    
        # http://localhost:5000/pythonlogin/admin/usagereports -report template page, edit the existing template
@app.route('/pythonlogin/admin/usagereports', methods=['GET', 'POST'])
def usage_reports():
    # Check if admin is logged-in
    if not admin_loggedin():
        return redirect(url_for('logout'))
    # Get the template directory path
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
  # Update the template file on save
    if request.method == 'POST':
        content = request.form['content'].replace('\r', '')
        open(template_dir + '/usagereports.html', mode='w', encoding='utf-8').write(content)
    # Read the activation email template
    content = open(template_dir + '/usagereports.html', mode='r', encoding='utf-8').read()
    return render_template('admin/usagereports.html', content=content)

# Dashboard Report page
@app.route('/pythonlogin/admin/sitereport')
def admin_sitereport():
    if not admin_loggedin():
        return redirect(url_for('logout'))


    # create context
    # try to get the year
    current_year = request.args.get('year', None)
    # Reset current year to present if year is None or not a 4 digit number
    if not current_year or not current_year.isdigit() or len(current_year.strip()) != 4:
        current_year = datetime.datetime.now().year
    else:
        current_year = int(current_year)
    
    context = {}
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # User who bought the book
    cursor.execute("""SELECT p.price AS plan, COUNT(i.price) AS total FROM planlist as p, invoice as i WHERE p.price=i.price AND i.status='paid' AND YEAR(i.updated_at)=%s GROUP BY plan""", (current_year,))
    today_result = cursor.fetchall()
    today_result = {p['plan']: p['total'] for p in today_result }

    # last month result
    last_month = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime(dt_format)
    cursor.execute("""SELECT p.price AS plan, COUNT(i.price) AS total FROM planlist as p, invoice as i WHERE p.price=i.price AND i.created_at < %s AND i.status='paid' GROUP BY plan""", (last_month,))
    last_month_result = cursor.fetchall()
    last_month_result = { p['plan']: p['total'] for p in last_month_result}

    totalplan = [35, 45, 55, 65]

    plan_status = {}

    for p in totalplan:
        today_total = today_result[p] if p in today_result else 0
        lastmonth_total = last_month_result[p] if p in last_month_result else 0
        if today_total >= lastmonth_total:
            textcolor = 'success'
            if today_total == 0:
                percent = 0
            else:
                if lastmonth_total == 0:
                    percent = today_total
                else:
                    percent = (today_total - lastmonth_total) / lastmonth_total
        
        else:
            textcolor = 'warning'
            percent = 0

        
        plan_status[p] = {
            'total' : today_total,
            'textcolor': textcolor,
            'percent' : percent
        }

    # total sale
    context['plan_status'] = plan_status
    total_sale = 0
    for p in plan_status:
        total_sale += p * plan_status[p]['total']

    
    context['total_sale'] = total_sale

    # total active user the membership is already purchased
    today = datetime.datetime.now().strftime(dt_format)
    cursor.execute('SELECT COUNT(*) FROM membership WHERE expire_at > %s', (today,))
    total_active = cursor.fetchone()['COUNT(*)']
    context['total_active'] = total_active

    # total sale group by month
    cursor.execute('''SELECT s.smonth AS smonth, COUNT(s.smonth) AS total FROM (
        SELECT MONTHNAME(created_at) AS smonth FROM invoice WHERE status=%s AND YEAR(created_at)=%s )
         AS s GROUP BY s.smonth''', ('paid',current_year,))
    sales_numbers = cursor.fetchall()
    sales_numbers = { p['smonth'][:3]: p['total'] for p in sales_numbers}
    # total emails group by month
    cursor.execute('''SELECT e.emonth AS emonth, COUNT(e.emonth) AS total FROM ( 
        SELECT MONTHNAME(promo_date) AS emonth from front_page_promo WHERE YEAR(promo_date)=%s)
        AS e GROUP BY e.emonth''',(current_year,))
    totals_email = cursor.fetchall()

    totals_email = {t['emonth'][:3]: t['total'] for t in totals_email} 

    # Total Number of Users Active

    monthlist = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_salelist = []
    monthly_emaillist =[]
    for m in monthlist:
        if m in totals_email:
            monthly_emaillist.append(totals_email[m])
        else:
            monthly_emaillist.append(0)

        if m[:3] in sales_numbers:
            monthly_salelist.append(sales_numbers[m])
        else:
            monthly_salelist.append(0)

    context['monthlist'] = monthlist
    context['monthly_salelist'] = monthly_salelist
    context['monthly_emaillist'] = monthly_emaillist
    context['current_year'] = current_year
    # year list queyr
    cursor.execute('''SELECT YEAR(updated_at) as year from invoice group by YEAR(updated_at)''')
    yearlist = cursor.fetchall()

    year_list = []
    for i in yearlist:
        year_list.append(i['year'])

    context['year_list'] = year_list
    print(context)

    # total number of guest user
    guest_qs = """SELECT g.gmonth as gmonth, COUNT(g.gmonth) as total FROM (SELECT MONTHNAME(updated_at) AS gmonth from pagevisit where is_login=%s AND YEAR(updated_at)=%s) AS g GROUP BY g.gmonth """

 
    # query the not login user
    cursor.execute(guest_qs, (0, current_year,))
    total_guest  = cursor.fetchall()
    total_guest = {g['gmonth'][:3]:g['total'] for g in total_guest }

    # an empty array
    monthly_guestlist = []
    for m in monthlist:
        if m in total_guest:
            monthly_guestlist.append(total_guest[m])
        else:
            monthly_guestlist.append(0)

    context['monthly_guestlist'] = monthly_guestlist

    return render_template('admin/reports.html', segment='index', context=context)

# Admin logged-in check function
def admin_loggedin():
    print("Admin check is called")
    if loggedin() and session['role'] == 'Admin':
        # admin logged-in
        return True
    # admin not logged-in return false
    return False
