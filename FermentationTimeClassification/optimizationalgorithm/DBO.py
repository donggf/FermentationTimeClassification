
import numpy as np
import random
import math
import copy
import time


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]

    return X, lb, ub


'''边界检查函数'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''蜣螂优化算法'''


def DBO(pop, dim, lb, ub, MaxIter, fun,b,k,rate):
    # 参数设置
    PballRolling = 0.2 # 滚球蜣螂比例
    PbroodBall = 0.4 #产卵蜣螂比例
    PSmall = 0.2 # 小蜣螂比例
    Pthief = 0.2 # 偷窃蜣螂比例
    BallRollingNum = int(pop*PballRolling) #滚球蜣螂数量
    BroodBallNum = int(pop*PbroodBall) #产卵蜣螂数量
    SmallNum = int(pop*PSmall) #小蜣螂数量
    ThiefNum = int(pop*Pthief) #偷窃蜣螂数量
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    # 记录全局最优
    minIndex = np.argmin(fitness)
    GbestScore = copy.copy(fitness[minIndex])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[minIndex, :])
    Curve = np.zeros([MaxIter, 1])
    Xl=copy.deepcopy(X) #用于记录X(t-1)
    # 记录当前代种群
    cX = copy.deepcopy(X)
    cFit = copy.deepcopy(fitness)
    Curve[0]=GbestScore
    for t in range(1,MaxIter):
        start_time = time.time()
        # print("第"+str(t)+"次迭代")
        # 蜣螂滚动 文献中式（1），（2）更新位置
        # 获取种群最差值
        maxIndex = np.argmax(fitness)
        Wort = copy.copy(X[maxIndex,:])
        r2 = np.random.random()
        for i in range(0,BallRollingNum):
            if r2<rate:
                if np.random.random()>0.5:
                    alpha=1
                else:
                    alpha=-1

                X[i,:]=cX[i,:]+b*np.abs(cX[i,:]-Wort)+alpha*k*Xl[i,:]
            else:
                theta = np.random.randint(180)
                if theta==0 or theta == 90 or theta == 180:
                    X[i,:]=copy.copy(cX[i,:])
                else:
                    theta = theta*np.pi/180
                    X[i,:]=cX[i,:]+np.tan(theta)*np.abs(cX[i,:]-Xl[i,:])
            for j in range(dim):
                if X[i,j]>ub[j]:
                    X[i,j]=ub[j]
                if X[i,j]<lb[j]:
                    X[i,j]=lb[j]
            fitness[i]=fun(X[i,:])
            if fitness[i]<GbestScore:
                GbestScore=copy.copy(fitness[i])
                GbestPositon[0,:]=copy.copy(X[i,:])
        # 当前迭代最优
        minIndex=np.argmin(fitness)
        GbestB = copy.copy(X[minIndex,:])
        # 蜣螂产卵 ，文献中式（3）
        R=1-t/MaxIter
        X1=GbestB*(1-R)
        X2=GbestB*(1+R)
        Lb = np.zeros(dim)
        Ub = np.zeros(dim)
        for j in range(dim):
            Lb[j]=max(X1[j],lb[j])
            Ub[j]=min(X2[j],ub[j])
        for i in range(BallRollingNum,BallRollingNum+BroodBallNum):
            b1=np.random.random()
            b2=np.random.random()
            X[i,:]=GbestB+b1*(cX[i,:]-Lb)+b2*(cX[i,:]-Ub)
            for j in range(dim):
                if X[i,j]>ub[j]:
                    X[i,j]=ub[j]
                if X[i,j]<lb[j]:
                    X[i,j]=lb[j]
            fitness[i]=fun(X[i,:])
            if fitness[i]<GbestScore:
                GbestScore=copy.copy(fitness[i])
                GbestPositon[0,:]=copy.copy(X[i,:])
        # 小蜣螂更新
        #文献中(5),(6)
        R=1-t/MaxIter
        X1=GbestPositon[0,:]*(1-R)
        X2=GbestPositon[0,:]*(1+R)
        Lb = np.zeros(dim)
        Ub = np.zeros(dim)
        for j in range(dim):
            Lb[j]=max(X1[j],lb[j])
            Ub[j]=min(X2[j],ub[j])
        for i in range(BallRollingNum+BroodBallNum,BallRollingNum+BroodBallNum+SmallNum):
            C1 = np.random.random([1,dim])
            C2 = np.random.random([1,dim])
            X[i,:]=GbestPositon[0,:]+C1*(cX[i,:]-Lb)+C2*(cX[i,:]-Ub)
            for j in range(dim):
                if X[i,j]>ub[j]:
                    X[i,j]=ub[j]
                if X[i,j]<lb[j]:
                    X[i,j]=lb[j]
            fitness[i]=fun(X[i,:])
            if fitness[i]<GbestScore:
                GbestScore=copy.copy(fitness[i])
                GbestPositon[0,:]=copy.copy(X[i,:])
        # 当前迭代最优
        minIndex=np.argmin(fitness)
        GbestB = copy.copy(X[minIndex,:])
        # 偷窃蜣螂更新
        # 文献中式（7）
        for i in range(pop-ThiefNum,pop):
            g=np.random.randn()
            S=0.5
            X[i,:]=GbestPositon[0,:]+g*S*(np.abs(cX[i,:]-GbestB)+np.abs(cX[i,:]-GbestPositon[0,:]))
            for j in range(dim):
                if X[i,j]>ub[j]:
                    X[i,j]=ub[j]
                if X[i,j]<lb[j]:
                    X[i,j]=lb[j]
            fitness[i]=fun(X[i,:])
            if fitness[i]<GbestScore:
                GbestScore=copy.copy(fitness[i])
                GbestPositon[0,:]=copy.copy(X[i,:])
        # 记录t代种群
        Xl= copy.deepcopy(cX)
        #更新当前代种群
        for i in range(pop):
            if fitness[i]<cFit[i]:
                cFit[i]=copy.copy(fitness[i])
                cX[i,:]=copy.copy(X[i,:])

        Curve[t] = GbestScore
        print(f'{t}',GbestScore,GbestPositon)
        end_time = time.time()
        print(f"Processing complete, elapsed time: {end_time - start_time:.2f} seconds.")
    return GbestScore, GbestPositon, Curve
