""" This program return hashed password based on secret provided """
import sys, hashlib
# You need to make sure app secret same as in main.py
app_secret = 'Skittles'

def get_hashpassword(password):
    hash = password + app_secret
    hash = hashlib.sha1(hash.encode())
    password = hash.hexdigest()
    return password


if __name__ == "__main__":
    if len(sys.argv) > 1:
        password = sys.argv[1]
        print(get_hashpassword(password))

    else:
        print('Syntax: python {}  <password>'.format(sys.argv[0]))

