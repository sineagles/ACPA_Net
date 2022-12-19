import xlsxwriter
import numpy as np
def write_PR(p,r,t,f1,filename):

    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    worksheet.activate()
    title = ['t','precision','recall','f1score']
    worksheet.write_row('A1',title)

     # Start from the first cell below the headers.
    n_row = 2
    for i in range(len(p)):
        insertData=[t[i],p[i],r[i],f1[i]]
        row = 'A' + str(n_row)
        worksheet.write_row(row, insertData)
        n_row=n_row+1
    workbook.close()

if __name__ == '__main__':
    p=np.array([1,2,3,4,5])
    r=np.array([2,3,4,5,6])
    t=np.array([0,1,2,3,4])

    write_PR(p,r,t,'pr.xlsx')