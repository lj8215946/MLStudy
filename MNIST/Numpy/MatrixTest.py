import numpy

w = numpy.matrix([[1, 2, 3], [1, 2, 4]])

b = numpy.matrix([[.1], [.2]])

a1 = numpy.matrix([[1], [3], [3]])

# b = numpy.matrix([[1, 2, 3], [4, 5, 6]])

z2 = numpy.add(numpy.dot(w, a1), b)

print(numpy.matrix(z2))

z2_ = - z2

wt = w.transpose()

print(numpy.dot(wt, z2_))
