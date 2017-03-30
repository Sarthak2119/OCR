def convert_ys(str1, str2):
    f=str1
    f=open(f,'r')
    f2=str2
    f2=open(f2,'w')
    f1=f.read()
    f1=f1.split('\n')
    for i in f1:
        x=0
        output_vec=[]
        print(i)
        if i>='A' and i<='Z':
            z=ord(i)-65
            x=10+z+1
        elif i>='a' and i<='z':
            z=ord(i)
            z-=98
            x=37+z+1
        elif i>='0' and i<='9':
            z=int(i)
            x=z+1
        else:
            continue

        # print (x)
        for i in range(1,63):
            if i==x:
                f2.write(str(1))
            else:
                f2.write(str(0))
            f2.write(' ')
        f2.write('\n')
    f.close()
    f2.close()

def run():
    convert_ys('test_out.txt', 'test_out2.txt')
    convert_ys('train_out.txt', 'train_out2.txt')