import os

def removefile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return 'remove file'
    else:
        return 'directory not found'
lst = os.listdir('D:/khu/inputData/Common/damagePolygon/processed')
path = 'D:/khu/inputData/Common/damagePolygon/processed/'
for i in lst:
    removefile(path + i)