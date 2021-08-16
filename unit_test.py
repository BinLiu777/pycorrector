import re

line = '你好！下午好！再见!明天见？好的?'
sentences = re.split('。|！|\!|\.|？|\?',line)
print(sentences)