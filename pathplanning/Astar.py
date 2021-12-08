import numpy as np
import math
import heapq
import copy
import matplotlib.pyplot as plt
import cv2 as cv

# Priority Queue based on heapq
class PriorityQueue:
    def __init__(self):
        self.elements = []
    def isEmpty(self):
        return len(self.elements) == 0
    def add(self, item, priority):
        heapq.heappush(self.elements,(priority,item))
    def remove(self):
        return heapq.heappop(self.elements)[1]
    def has(self, elem):
        return elem in self.elements

def getStartGoal(map):
    startX = 0
    startY = 0
    goalX = 0
    goalY = 0   
    for y in range(len(map)):
        for x in range(len(map[0])):
            if map[y][x] == -2:
                startX = x
                startY = y
            if map[y][x] == -3:
                goalX = x
                goalY = y
    return (startX,startY),(goalX,goalY)        
    
       
def get_neighbors(map, current):
    x,y = current
    returnNeighbors = []
    if(x != 0 and map[x-1][y] != 255):
        returnNeighbors.append((x-1,y))
    if(x < len(map)-1 and map[x+1][y] != 255):
        returnNeighbors.append((x+1,y))
        
    if(y < len(map[0])-1 and map[x][y+1] != 255):
        returnNeighbors.append((x,y+1))
        
    if(y != 0 and map[x][y-1] != 255):
        returnNeighbors.append((x,y-1))
    return returnNeighbors
    
def manhattanDistance(start, next):
    startX, startY = start
    nextX, nextY = next
    return abs(startX - nextX) + abs(startY - nextY)

def euclidesDistance(start, next):
    startX, startY = start
    nextX, nextY = next
    return math.sqrt((startX - nextX) * (startX - nextX) + (startY - nextY) * (startY - nextY))

def createPath(came_from, start, goal):
    current = goal
    path = np.array(list(goal))
    while(current != start):
        current = came_from[current]
        path = np.append(path, list(current))
    
    path = path.reshape(int(len(path)/2), 2)
    
    return path
# An example of search algorithm, feel free to modify and implement the missing part
def search(map, start, goal, algorithm = "eucliden" ,ytop = None, ybot = None, minx = None):
    #cost_function = euclidesDistance
    cost_function = None
    if(algorithm == "manhattan"):
        print("manhattan")
        cost_function = manhattanDistance
    elif (algorithm == "eucliden"):
        print("euclides")
        cost_function = euclidesDistance
  
    # open list
    frontier = PriorityQueue()

    # add starting cell to open list
    frontier.add(start, 0)

    # path taken
    came_from = {}

    gScore = {}
    gScore[start] = 0

    while not frontier.isEmpty():
        
        current = frontier.remove()
        if current == goal:
            print("Goal reached")
            x,y = current
            map[x][y] = -3
            return createPath(came_from, start, current)
        for next in get_neighbors(map, current):
            x,y = next

            tempCost = gScore[current] + 1
            if(tempCost < gScore.get(next, 999)):
                came_from[next] = current
                gScore[next] = tempCost
                map[x][y] = tempCost
                
                if(not frontier.has(next)):        
                    fScore = gScore[next] + cost_function(next, goal)
                    frontier.add(next, fScore)
    return {}

def main():
   ob = cv.imread("miniMap.png", 0)
   thresh = 127
   tm, ob = cv.threshold(ob, thresh, 255, cv.THRESH_BINARY)
   shape = ob.shape
   m = copy.deepcopy(ob)
   ox, oy = [], []
   for x in range(ob.shape[0]):
       for y in range(ob.shape[1]):
           if(ob[x,y] == 0):
               m[x,y] = 0
           else:
               ox.append(x)
               oy.append(y)
   print(m)
   show_animation = True
   home = (150, 1190)
   homeStart = (150, 1090)
   neighbour = (790, 1100)
   roundabout = (370, 850)

   miniStartHome = (15, 110) #(x, y) format
   miniHalfRound = (50 , 5)
   miniNeighbour = (90 , 108)
   miniFirstQuarter = (10,46)
   miniLastQuarter = (91,45)
   miniEndHome = (15,119)
   wayPoint1 = (10,45)
   wayPoint2=(48,5)
   wayPoint3 = (90,45)
   start = miniStartHome
   goal = miniNeighbour

   if show_animation:
       plt.plot(ox, oy, ".k")
       plt.plot(start[0], start[1], "og")
       plt.plot(goal[0], goal[1], "xb")
       plt.axis("equal")

   #route = search(m, start, goal)
   #print(route)
   route1 = search(m, start, wayPoint1)
   route2 = search(m, wayPoint1, wayPoint2)
   route3 = search(m, wayPoint2, wayPoint3)
   route4 = search(m, wayPoint3, start)

   route = np.concatenate((route4, route3))
   route = np.concatenate((route, route2), axis=0)
   route = np.concatenate((route, route1), axis=0)

   if show_animation:
       plt.scatter(route[:,0], route[:,1], marker=".", c="red")
       plt.show()

if __name__ == '__main__':
    main()
