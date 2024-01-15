from operator import itemgetter
import sys


min_support,input_,output_=float(sys.argv[1]),sys.argv[2],sys.argv[3]

addr_dict={}
addr_list=[] #符合的Data寫入list，以利未來存入output.txt
ans_list=[] #將input以list方式儲存
raw=0 #input列數
count=1 #計算第二輪之後每組frequent pattern的Data個數

def write_to_output(list_):
    """ 將此輪結果寫進output.txt"""
    with open(output_,'a') as f:
        for i in list_:
            str_=""
            for j in i[0]:
                str_+=j+","
            str_=str_.rstrip(",")
            f.write(f"{str_}:{round(i[1]/raw,4):.4f}\n")

def sort_the_list(list_,raw):
    """將符合資格的Data排序"""
    try:
        for i in range(1,-1,-1):
            list_.sort(key = lambda s: int(s[0][i]))
    except:
        return list_  
    return list_

# 第一round，讀入input並跑出第一輪結果
with open(input_, 'r') as f:
    for line in f:
        ans_list.append(line.rstrip().split(','))
        raw+=1
        line=line.rstrip()
        for i in line.split(','):
            i=int(i)
            if i in addr_dict:
                addr_dict[i]+=1
            else:
                addr_dict[i]=1
for key,value in addr_dict.items():
    addr_list.append([[str(key)],value])
if addr_list:tag=1


addr_list=[x for x in addr_list if x[1]/raw>min_support]
addr_list.sort(key = lambda s: int(s[0][0]))
write_to_output(addr_list)


#第二輪以後，沿用上一輪結果並跑出下一次結果
while tag==1:
    count+=1
    tag=0
    temp=[]
    addr_dict={}
    for i in range(len(addr_list)):
        for j in range(i+1,len(addr_list)):
            if addr_list[i][0][:-1]==addr_list[j][0][:-1]:
                temp.append(addr_list[i][0]+[addr_list[j][0][-1]])
    for line in ans_list:
        for i in temp:
            result = [word for word in i if str(word) in line]
            if result==i:
                str_=""
                for i in result:
                    str_+=f"{i},"
                str_=str_.rstrip(",")
                if str_ in addr_dict:
                    addr_dict[str_]+=1
                else:
                    addr_dict[str_]=1
    addr_list=[]
    for key,value in addr_dict.items():
        addr_list.append([key.split(","),value])
    addr_list=[x for x in addr_list if x[1]/raw>=min_support]
    addr_list=sort_the_list(addr_list,count)
    if addr_list:
        tag=1  #還有下一round則tag==1
    write_to_output(addr_list)