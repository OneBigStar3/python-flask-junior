import os
import time

path = "/home/ec2-user/site/static/uploads/cover_uploads/"
now = time.time()

for filename in os.listdir(path):
  print(f"Removing {filename}")
  os.remove(os.path.join(path, filename))