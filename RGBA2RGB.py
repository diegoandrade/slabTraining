from PIL import Image
import os
src = "./resizedData"
dst = "./resized_black/"

dst = "/resizedData" # resized
path = os.getcwd() + dst
print (path)

# define the name of the directory to be created
dst1 = "/resized_black" # resized
path1 = os.getcwd() + dst1

try:
    os.mkdir(path1)
except OSError:
    print ("Creation of the directory %s failed" % path1)
else:
    print ("Successfully created the directory %s " % path1)

for each in os.listdir(src):
    path3 = path + "/" + each
    #print (path3)
    png = Image.open(path3)

    if png.mode == 'RGBA':
        png.load() # required for png.split()
        background = Image.new("RGB", png.size, (0,0,0))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        background.save(os.path.join(path1,each.split('.')[0] + '.jpg'), 'JPEG')
    else:
        #print("HOLA2")
        png.convert('RGB')
        outp = each.split('.')[0] + '.jpg'
        png.save(os.path.join(path1,each.split('.')[0] + '.jpg'), 'JPEG')
        #png.save(outp, "JPEG", quality=80, optimize=True, progressive=True)
