import spacy
import mysql.connector
import pandas as pd


nlp = spacy.load('en_core_web_lg')



def get_similar(description,string_2):
    pass
    




def read_input():
    
    cnx = mysql.connector.connect(user='root', password='StreamDeck7692$',
                              host='localhost',
                              database='book_db')

    try:
        cursor = cnx.cursor()
        cursor.execute("""
            select * from book_metadata
        """)
        result = cursor.fetchall()
        return result
    finally:
        cnx.close()

    

# print(get_similar("Hello","shani"))

def file_write(sv_res):
    with open("result_.txt",'w',encoding='utf-8') as f:
        f.write("Percentage,title,ISBN,Description")
        for a in sv_res:
            f.write(str(a) + "\n")


res = read_input()
sv_res = []
input_ = "Charlie Reade looks like a regular high school kid, great at baseball and football, a decent student. But he carries a heavy load. His mom was killed in a hit-and-run accident when he was ten, and grief drove his dad to drink. Charlie learned how to take care of himself—and his dad. When Charlie is seventeen, he meets a dog named Radar and her aging master, Howard Bowditch, a recluse in a big house at the top of a big hill, with a locked shed in the backyard. Sometimes strange sounds emerge from it."
print("data_len: "+str(len(res)))
print(res[4030])


res_ = []
for r in res:
    res_.append(str(r[2]))
processed_docs_1 = nlp.pipe(res_)
processed_docs_2 = nlp(input_)

for i in range(len(res)):
    sim_check = str(res[i][2])
    s_1 = next(processed_docs_1)
    
   # print(i)
    
    s = s_1.similarity(processed_docs_2)
    

    c = [str(s),str(res[i][1]),str(res[i][12]),str(res[i][2])]
    sv_res.append(c)
    #print(s)



# file_write(sv_res)

#print(sv_res)

df = pd.DataFrame(list(map(list,set(map(tuple,sv_res)))),columns=["percentage","title","isbn","description"])
cc = df.sort_values(["percentage"], ascending=False).head(4)


cc.to_csv("res.csv",index=False)








