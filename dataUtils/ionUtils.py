
import numpy as np
import pandas as pd

def getFullScoresTest(data,thr,BR=range(12)):
    
    result=np.array([])
    INDICES=np.array([])
    df = pd.DataFrame(data)
    for b in BR:
        THR=thr[b]
        
        batch_ixs = getTestRangesIx()[b]
        
        ################################### thr #################################################
        batch=df.loc[batch_ixs]
        probs=batch.values[:,3:14]
        
        batch=batch[(np.max(probs,axis=1)>=THR)]
        probs=batch.values[:,3:14]
        
        batch["pred"]=np.argmax(probs,axis=1)
        pred=batch["pred"].values
        indices=batch["id"].values
        
        
        #-------------------------------------------------------------------------------------------
        batch=df.loc[batch_ixs]
        probs=batch.values[:,3:14]
        
        batch=batch[(np.max(probs,axis=1)<THR)]
        probs=batch.values[:,3:14]*batch.values[:,15:]
        
        batch["pred"]=np.argmax(probs,axis=1)
        
        pred=np.concatenate((pred,batch["pred"].values))
        indices=np.concatenate((indices,batch["id"].values))
        
        
        result=np.concatenate((result,pred))
        INDICES=np.concatenate((INDICES,indices))
    return result.astype(int),INDICES.astype(int)




from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def getFullScores(data,thr=0.63,BR=range(11)):
    PRES="%.6f"
    
    result={}
    #df = pd.read_csv(file)
    df = pd.DataFrame(data)
    for b in BR:
        batch_ixs = getTrainRangesIx()[b]
        
        ################################### thr #################################################
        batch=df.loc[batch_ixs]
        probs=batch.values[:,4:15]
        
        batch=batch[(np.max(probs,axis=1)>=thr)]
        probs=batch.values[:,4:15]
        
        batch["pred"]=np.argmax(probs,axis=1)
        
        true=batch["open_channels"].values
        pred=batch["pred"].values
        
        #-------------------------------------------------------------------------------------------
        batch=df.loc[batch_ixs]
        probs=batch.values[:,4:15]
        
        batch=batch[(np.max(probs,axis=1)<thr)]
        
        probs=batch.values[:,4:15]*batch.values[:,16:]
        batch["pred"]=np.argmax(probs,axis=1)
        
        true=np.concatenate((true,batch["open_channels"].values))
        pred=np.concatenate((pred,batch["pred"].values))
        
        
        f1=f1_score(true, pred, average='macro')
        acc=accuracy_score(true, pred)
        result={"thr":"%.3f" % thr, "batch":b,"acc":PRES % acc,"f1":PRES % f1}
        
        
    return result,true,pred



from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#def getScores(file,thr=[0,1],BR=range(11)):
def getScores(data,thr=[0,1],BR=range(11)):
    PRES="%.6f"
    
    result=[]
    #df = pd.read_csv(file)
    df = pd.DataFrame(data)
    for b in BR:
        batch_ixs = getTrainRangesIx()[b]
        
        
        ################################### thr #################################################
        batch=df.loc[batch_ixs]
        
        probs=batch.values[:,4:15]
        batch=batch[(np.max(probs,axis=1)>=thr[0]) & (np.max(probs,axis=1)<=thr[1])]
        
        probs=batch.values[:,4:15]
        batch["pred"]=np.argmax(probs,axis=1)
        f1=f1_score(batch["open_channels"], batch["pred"], average='macro')
        acc=accuracy_score(batch["open_channels"], batch["pred"])
        result.append({"thr":{"batch":b,"acc":PRES % acc,"f1":PRES % f1}})
        
        
        
        #####################################  seq  #############################################
        
        batch=df.loc[batch_ixs]
        probs=batch.values[:,4:15]
        batch=batch[(np.max(probs,axis=1)>=thr[0]) & (np.max(probs,axis=1)<=thr[1])]
        
        probs=batch.values[:,16:]
        batch["pred"]=np.argmax(probs,axis=1)
        f1=f1_score(batch["open_channels"], batch["pred"], average='macro')
        acc=accuracy_score(batch["open_channels"], batch["pred"])
        
        result.append({"seq":{"batch":b,"acc":PRES % acc,"f1":PRES % f1}})
        
        
        
        #####################################  thr*seq  #############################################
        
        batch=df.loc[batch_ixs]
        probs=batch.values[:,4:15]
        batch=batch[(np.max(probs,axis=1)>=thr[0]) & (np.max(probs,axis=1)<=thr[1])]
        
        probs=batch.values[:,4:15]*batch.values[:,16:]
        batch["pred"]=np.argmax(probs,axis=1)
        f1=f1_score(batch["open_channels"], batch["pred"], average='macro')
        acc=accuracy_score(batch["open_channels"], batch["pred"])
        
        result.append({"thr*seq":{"batch":b,"acc":PRES % acc,"f1":PRES % f1}})
        
        
        
    return result


def getTestRangesIx():
    test_ranges={}
    for i in range(12):
        if i<10:
            l=100000 #length
            s= i*l   #start
            test_ranges[i]=np.arange(s,s+l)
        else:
            s+=l
            l=500000
            test_ranges[i]=np.arange(s,s+l)
    return test_ranges

def getTrainRangesIx():    
    train_ranges={}
    s0=0
    for i in range(11):
        l=500000 #length
        if(i==1):l=100000
        if(i==2):l=400000
        train_ranges[i]=np.arange(s0,s0+l)
        s0+=l
    return train_ranges

def getTrainRanges():
    return np.arange(0,550,50)

def getTestRanges():
    testTR=list(np.arange(500,601,10))+[650,700]
    return np.array(testTR)





#Plotting
#interactive plot dots
def plotSC2(x,y):
    import plotly.express as px
    fig = px.scatter(x=x, y=y)
    fig.show()
    