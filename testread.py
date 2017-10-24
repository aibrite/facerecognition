txsst = open('links/negative_urls.txt', encoding='utf-8')

# for l in txsst.readlines():
#     line = l[:l.find('\n')]
#     print(line)

a = txsst.read()
b = a.split('\n')
print(a)
