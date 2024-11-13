with open('./aa.csv', 'r') as inFp:
    with open('./aa2.csv', 'w') as outFp:
        header = inFp.readline()
        header = header.strip()
        header_list = header.split(',')

        idx1 = header_list.index('ID')
        idx2 = header_list.index('Name')
        idx3 = header_list.index('Averge Height')
        header_list = [header_list[idx1], header_list[idx2], header_list[idx3]]
        outFp.write(','.join(map(header_list))+'\n')