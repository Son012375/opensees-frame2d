import sys, io, fitz
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
doc = fitz.open(r'C:\Users\youm\Downloads\research_proposal_baek.pdf')
print(f'Pages: {len(doc)}')
for i in range(len(doc)):
    print(f'\n===PAGE {i+1}===')
    print(doc[i].get_text())
doc.close()
