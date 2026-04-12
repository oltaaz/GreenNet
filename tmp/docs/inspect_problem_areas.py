from docx import Document
path='/Users/oltazagraxha/Desktop/GreenNet/output/doc/RIT_Digital_Institutional_GreenNet_reorganized.docx'
doc=Document(path)
for start,end in [(48,63),(73,79),(147,153),(164,175)]:
    print('\nRANGE',start,end)
    for i in range(start,end):
        p=doc.paragraphs[i]
        print(f'{i:03d} [{p.style.name}] {p.text}')
