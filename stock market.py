from tkinter import *
from PIL import ImageTk,Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas_datareader import data as dt
import datetime
from tkinter import messagebox
import matplotlib.pyplot as plt

#MAIN SCREEN
scr=Tk()
scr.title("Stock Market")
scr.geometry('500x500')
stack_label=Label(scr,text="STOCK NAME:",font=("calibri",12,"bold"))
stack_label.place(x=20,y=105)
stock=StringVar()
l=["googl","amzn","aapl","msft"]
stack_menu=OptionMenu(scr,stock,*l)
stack_menu.place(x=128,y=100)
stock.set(l[0])
test_label=Label(scr,text="TEST SIZE :",font=("calibri",12,"bold"))
test_label.place(x=30,y=130)
testsize=StringVar()
lt=[0.1,0.2,0.3,0.4]
test_menu=OptionMenu(scr,testsize,*lt)
test_menu.place(x=120,y=130)
l=Label(scr,text="%",font=("times",20,"bold"))
l.place(x=178,y=130)
testsize.set(lt[0])
eval_label=Label(scr,text="CLICK TO PREDICT :",font=("calibri",16,"bold"))
eval_label.place(x=23,y=240)
q2 = lambda: predict1(pr)
eval_button=Button(scr,text='RESULT',height='2',width='30',activebackground="cyan",activeforeground="green",bd=3,bg="lavender",fg="blue",justify="right",relief="raised",command=q2)
eval_button.place(x=190,y=240)
acc_label=Label(scr,text="ACCURACY :",font=("calibri",16,"bold"))
acc_label.place(x=23,y=300)
q1 = lambda: linREGG(x,y,x1,x_pr)
acc_button=b=Button(scr,text='EVALUATE',height='2',width='30',activebackground="cyan",activeforeground="green",bd=3,bg="lavender",fg="blue",justify="right",relief="raised",command=q1)
acc_button.place(x=168,y=300)
vis_label=Label(scr,text="DATA CHART :",font=("calibri",16,"bold"))
vis_label.place(x=32,y=360)
q3 = lambda:  graph(f,alg)
vis_button=b=Button(scr,text='VISUALISE',height='2',width='30',activebackground="cyan",activeforeground="green",bd=3,bg="lavender",fg="blue",justify="right",relief="raised",command=q3)
vis_button.place(x=168,y=360)
sp=IntVar()
predict_label=Label(scr,text="ENTER NUMBER OF DAYS :",font=("calibri",16,"bold"))
predict_label.place(x=23,y=180)
predict_entry=Entry(scr,textvariable=sp,bd=10,font=('calibri',12,'bold'))
predict_entry.place(x=260,y=170)
sp.set(20)
days=sp.get()

#DATA 
def fetch_data():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime.now()
    f = dt.DataReader("msft", 'tiingo', start, end,access_key='911ee28d70118f9cea5a84d2b8f1436fa32d3116')
    return f

def google():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime.now()
    f= dt.DataReader('googl', 'tiingo', start, end,access_key='304d70c66c3a1061ac4decb79cfad9034fed26d7')
    return f


def aapl():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime.now()
    f= dt.DataReader('aapl', 'tiingo', start, end,access_key='304d70c66c3a1061ac4decb79cfad9034fed26d7')
    return f
 
def amzn():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime.now()
    f= dt.DataReader('amzn', 'tiingo', start, end,access_key='304d70c66c3a1061ac4decb79cfad9034fed26d7')
    return f

f=fetch_data()


#ACCURACY USING LINEAR REGRESSION
def linREGG(x,y,x1,x_pr):
    x_tr,x_ts,y_tr,y_ts=train_test_split(x1,y,test_size=0.2)
    alg=LinearRegression()
    alg.fit(x_tr,y_tr)
    score=alg.score(x_ts,y_ts)
    messagebox.showinfo("accuracy","YOR ACCURACY IS = "+str(score))
    return alg.predict(x_pr)

def linREGG12(x,y,x1,x_pr):
    x_tr,x_ts,y_tr,y_ts=train_test_split(x1,y,test_size=0.2)
    alg=LinearRegression()
    alg.fit(x_tr,y_tr)
    score=alg.score(x_ts,y_ts)
    return alg.predict(x_pr),alg




def data(f,days):
    f.reset_index(inplace=True)
    f.set_index('date',inplace=True)
    f=f[['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume',]]
    no_days=days
    pd.set_option('mode.chained_assignment', None)
    f.loc[:,'newclose']=f.loc[:,'adjClose'].shift(-no_days)
    x=f.drop(['adjClose','newclose'],axis=1)
    y=f['newclose'].dropna()
    x1=x[:-no_days]
    x_pr=x[-no_days:]
    return x,y,x1,x_pr
#PREDICTION
def predict1(pr):
    new=Toplevel(scr)
    new.title("Prediction")
    new.geometry("500x200")
    l=Label(new,text=str(pr),font=('calibri',12,'bold'))
    l.place(x=5,y=60)


x,y,x1,x_pr=data(f,days)
pr,alg=linREGG12(x,y,x1,x_pr)
   

#GRAPH
          
def graph(f,alg):
    f.reset_index(inplace=True)
    f.set_index('date',inplace=True)
    f=f[['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume',]]
    no_days=20
    pd.set_option('mode.chained_assignment', None)
    f.loc[:,'newclose']=f.loc[:,'adjClose'].shift(-no_days)
    f['forecast']=np.nan
    prd=alg.predict(x_pr)
    lastday=(f.iloc[-1].name)
    for i in prd:
        lastday+=datetime.timedelta(1)
        f.loc[lastday]=[np.nan for _ in range(6)]+[i]
    f['adjClose'].plot()
    f['forecast'].plot()
    plt.show()
























