from daal.data_management import HomogenNumericTable, BlockDescriptor_Float64, readOnly

def getArrayFromNT(table, num_rows=0):
    bd = BlockDescriptor_Float64()
    if num_rows == 0:
        num_rows = table.getNumberOfRows()
    table.getBlockOfRows(0, num_rows, readOnly, bd)
    npa = bd.getArray()
    table.releaseBlockOfRows(bd)
    return npa

def printNT(table, num_printed_rows = 0, message=''):
    npa = getArrayFromNT(table, num_printed_rows)
    print(message, '\n', npa)

