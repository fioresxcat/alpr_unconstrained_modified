import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open('1.txt', 'r') as fp:
    for line in fp:
        data 		= line.strip().split(',')
        ss 			= int(data[0])   # là số đỉnh, luôn là 4
        lptype      = int(data[1])  ### fix
        values 		= data[2:(ss*2 + 2)]
        text 		= data[(ss*2 + 2)] if len(data) >= (ss*2 + 3) else ''
        break

print(f'data: {data}')
print(f'ss: {ss}')
print(f'lptype: {lptype}')
print(f'values: {values}')
print(f'text: {text}')