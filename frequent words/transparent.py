from PIL import Image
  
def convertImage(PATH):
    img = Image.open(PATH)
    img = img.convert("RGBA")
  
    datas = img.getdata()
  
    newData = []
  
    for item in datas:
        if item[0] >= 245 and item[1] >= 245 and item[2] >= 245:
            newData.append((item[0], item[1], item[2], 0))
        elif item[0] <= 10 and item[1] <= 10 and item[2] <= 10:
            newData.append((item[0], item[1], item[2], 0))
        else:
            newData.append(item)
  
    img.putdata(newData)
    img.save(PATH, "PNG")
    print("Successful")