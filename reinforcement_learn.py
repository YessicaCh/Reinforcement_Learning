import numpy as np

points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
goal = 12

# how many points in graph? x points
MATRIX_SIZE = 16+1

# create matrix x*y
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
R *= -2

R[1,1]=1
R[1,2]=1
R[2,2]=0
R[2,1]=1
R[2,3]=1
R[3,2]=1
R[3,4]=1
R[3,3]=0
R[4,8]=1
R[4,4]=0
R[8,4]=1
R[8,7]=1
R[8,12]=1
R[8,8]=0
R[7,8]=1 #I
R[7,6]=-1 #D
R[7,11]=1 #
R[7,7]= 0
R[6,6]=0
R[6,7]= 1 #I
R[6,5]= -1 #D
R[6,10]= 1 #  A
R[5,5]= -5 #  A
R[5,6]= -5 #I
R[5,9] = -5 #A
R[9,5]= 1 # Arriba
R[9,9]= 0 #
R[9,10] = 1 #I   
R[9,13] = 1 #ABajo  
R[11,7] = 1 #A  
R[11,12] = 1 #I
R[11,10] = -1 #D
R[11,11] = 0 #I
R[10,6] = 1 #Ar
R[10,9] = -1 #D
R[10,11] = 1 #I
R[10,10] = 0 #I
R[12,12] = 5 #
R[12,8] = 5 # AR
R[12,11] = -5 #D
R[13,13] = 0 #I
R[13,9] = 1 #Ar
R[13,14] = 1 #I
R[14,13] = -1 #D
R[14,15] = 1 #I
R[14,14]= 0
R[15,15] = 0
R[15,16] = 1 #  I
R[15,14] = -1 # D
R[16,16] = 0
R[16,15] = 1 #



#print(R)
# assign zeros to paths and 100 to goal-reaching point
"""
for point in points_list:
    print(point)
    if point[1] == goal:
          R[point] = 100
    else:
          R[point] = 0

    if point[0] == goal:
          R[point[::-1]] = 100
    else:
          # reverse of point
          R[point[::-1]]= 0
    #print(R)
    #print("\n")
"""
#print(R)
# add goal point round trip
#R[goal,goal]= 100
#print(R)
Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

# learning parameter
gamma = 1

initial_state = 2


def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

available_act = available_actions(initial_state) 



def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action

action = sample_next_action(available_act)

def update(current_state, action, gamma):
    
  max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
  
  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  
  Q[current_state, action] = R[current_state, action] + gamma * max_value
  print('max_value', R[current_state, action] + gamma * max_value)
  
  if (np.max(Q) > 0):
    return(np.sum(Q/np.max(Q)*100))
  else:
    return (0)
    
update(initial_state, action, gamma)



# Training
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    print("current_state",current_state)
    available_act = available_actions(current_state)
    print("available_act",available_act)
    action = sample_next_action(available_act)
    print("action",action)
    score = update(current_state,action,gamma)
    scores.append(score)
    print ('Score:', str(score))
    print("\n")
    
print("Trained Q matrix:")
print(Q/np.max(Q)*100)


# Testing
current_state = 0
steps = [current_state]

while current_state != 7:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    print(next_step_index)
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)
print("0      1    2    3    4   5   6   7 ")
print(R)

#plt.plot(scores)
#plt.show()
#Most efficient path:
#[0, 1, 2, 7]

