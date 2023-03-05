#!/usr/bin/python
import sys
sys.path.insert(0, '/home/ec2-user/site/venv/')
sys.path.append('/home/ec2-user/site/venv/lib/python3.8/site-packages')


#from __init__ import app as application

from site import app

if __name__ == "__main__":
    app.run()
