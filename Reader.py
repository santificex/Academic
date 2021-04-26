import PyPDF2
import time 
filename = 'C:/Users/santi/GitHub/Academic/InstLeadership.pdf'
pdfFileObj = open(filename,'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
num_pages = pdfReader.numPages
count = 0
text = ""
print("First word to start reading: ")
inicio=int(input())
print("Desired Words per minute: ")
velocidad=int(input())
vel=60/velocidad
while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()
if text != "":
   text = text
else:
   text = textract.process(fileurl, method='tesseract', language='eng')
texto=text.split()
contador=0
for i in range(inicio,len(texto)-1):
   if '.' in texto[i]:
      print(texto[i].ljust(80)+str(contador))
      time.sleep(vel)
      time.sleep(0.8)
      contador+=1
   elif ',' in texto[i]:
      print(texto[i].ljust(80)+str(contador))
      time.sleep(vel)
      time.sleep(0.3)
      contador+=1
   else:
      print(texto[i].ljust(80)+str(contador))
      time.sleep(vel)
      contador+=1
