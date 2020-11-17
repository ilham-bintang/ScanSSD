import os


root = "GTDB/images"
result = []

for dirname in os.listdir(root):
    for filename in os.listdir(os.path.join(root, dirname)):
        result.append(dirname+"/"+filename)

with open('result.txt', 'w') as f:
    for item in result:
        f.write("%s\n" % item)