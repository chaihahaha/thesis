import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Unifont'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def readlist(filename):
    with open(filename, "r") as f:
        s = f.read()
    l = [float(i) for i in s.split("\n") if isfloat(i) ]
    return l

fcn_reward = readlist("fcn_reward.txt")
fcn_lsh_reward = readlist("fcn_lsh_reward.txt")
fcn_sim_reward = readlist("fcn_sim_reward.txt")
fcn_lossmu = readlist("fcn_lossmu.txt")
fcn_lossq1 = readlist("fcn_lossq1.txt")
fcn_lossq2 = readlist("fcn_lossq2.txt")
fcn_suc_rate = readlist("fcn_suc_rate.txt")

nofcn_reward = readlist("nofcn_reward.txt")
nofcn_lsh_reward = readlist("nofcn_lsh_reward.txt")
nofcn_sim_reward = readlist("nofcn_sim_reward.txt")
nofcn_lossmu = readlist("nofcn_lossmu.txt")
nofcn_lossq1 = readlist("nofcn_lossq1.txt")
nofcn_lossq2 = readlist("nofcn_lossq2.txt")
nofcn_suc_rate = readlist("nofcn_suc_rate.txt")

epoch = [i+1 for i in range(min(len(fcn_lossmu),len(nofcn_lossmu)))]
fig, ax = plt.subplots()
ax.plot(epoch, fcn_reward[:len(epoch)],'r', label='浅层网络')
ax.plot(epoch, nofcn_reward[:len(epoch)],'b',label='深层网络')
ax.set_title('平均环境奖励')
ax.set_xlabel('Epoch')
ax.set_ylabel('Reward')
ax.legend()
fig.savefig("myenv_reward.png")
fig.show()

fig, ax = plt.subplots()
ax.plot(epoch, fcn_lsh_reward[:len(epoch)],'r', label='浅层网络')
ax.plot(epoch, nofcn_lsh_reward[:len(epoch)],'b',label='深层网络')
ax.set_title('平均LSH内部探索奖励')
ax.set_xlabel('Epoch')
ax.set_ylabel('Reward')
ax.legend()
fig.savefig("myenv_lsh_reward.png")
fig.show()

fig, ax = plt.subplots()
ax.plot(epoch, fcn_sim_reward[:len(epoch)],'r', label='浅层网络')
ax.plot(epoch, nofcn_sim_reward[:len(epoch)],'b',label='深层网络')
ax.set_title('平均物理引擎仿真时间奖励')
ax.set_xlabel('Epoch')
ax.set_ylabel('Reward')
ax.legend()
fig.savefig("myenv_sim_reward.png")
fig.show()

fig, ax = plt.subplots()
ax.plot(epoch, fcn_lossmu[:len(epoch)],'r', label='浅层网络')
ax.plot(epoch, nofcn_lossmu[:len(epoch)],'b',label='深层网络')
ax.set_title('平均Actor损失')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
fig.savefig("myenv_lossmu.png")
fig.show()

fig, ax = plt.subplots()
ax.plot(epoch, fcn_lossq1[:len(epoch)],'r', label='浅层网络')
ax.plot(epoch, nofcn_lossq1[:len(epoch)],'b',label='深层网络')
ax.set_title('平均Critic1损失')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
fig.savefig("myenv_lossq1.png")
fig.show()

fig, ax = plt.subplots()
ax.plot(epoch, fcn_lossq2[:len(epoch)],'r', label='浅层网络')
ax.plot(epoch, nofcn_lossq2[:len(epoch)],'b',label='深层网络')
ax.set_title('平均Critic2损失')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
fig.savefig("myenv_lossq2.png")
fig.show()

fig, ax = plt.subplots()
ax.plot(epoch, fcn_suc_rate[:len(epoch)],'r', label='浅层网络')
ax.plot(epoch, nofcn_suc_rate[:len(epoch)],'b',label='深层网络')
ax.set_title('平均成功率')
ax.set_xlabel('Epoch')
ax.set_ylabel('Success rate')
ax.legend()
fig.savefig("myenv_suc_rate.png")
fig.show()
