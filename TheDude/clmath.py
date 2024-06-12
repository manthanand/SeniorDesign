def pointwise_subtraction(list1, list2):
    list3 = []
    for i in range(len(list1)):
        list3.append(list1[i] - list2[i])

    return list3

def pointwise_addition(list1, list2):
    list3 = []
    for i in range(len(list1)):
        list3.append(list1[i] + list2[i])

    return list3