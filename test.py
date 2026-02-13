from modules.matrix import Matrix
from modules.tools import Datas

if __name__ == "__main__":
    ### matrix test
    """
    m = Matrix([[1,2,3], [4,5,6]])
    transpose = m.T
    m2 = Matrix([[2,2,2], [2,2,2]])
    m3 = Matrix([[10,11], [12,13], [14,15]])

    print(f"m:\n{m}")
    print(f"m transpose:\n{transpose}")
    print(f"m2:\n{m2}")
    print(f"add m+m2:\n{m.add(m2)}")
    print(f"m3:\n{m3}")
    print(f"mul m.m3:\n{m.mul(m3)}")
    """
    ### datas test
    my_datas = Datas.load_csv("inputs/data.csv", "linear_regression")
    my_datas.set_features(["x1"])
    my_datas.set_target("y")
    """
    print(my_datas.names)
    print(my_datas.values)
    print(my_datas.datas_table)
    print(my_datas.m)
    print(my_datas.n)
    print(my_datas.features)
    print(my_datas.target)
    print(my_datas.get_polynomial_expression_result(0, 1, [1]))
    y_calculated = []
    y_init = 2 # what to be found using gradient descent
    coef = [1] # what's to be found using gradient descent
    for x in my_datas.features["x1"]:
        y_calculated.append(my_datas.get_polynomial_expression_result(y_init, x, coef))
    print(y_calculated)
    print(my_datas.MSE(my_datas.target["y"], y_calculated))
    """
    
    ### find minimum RSE, so best parameters
    # start with arbitrary set parameters
    a = 53
    b = 6
    y_calculated = []
    for x in my_datas.features["x1"]:
        y_calculated.append(my_datas.get_polynomial_expression_result(a, x, [b]))
    # get first MSE
    current_mse = my_datas.MSE(my_datas.target["y"], y_calculated)
    previous_mse = current_mse
    # set a step for gradient descent algorithm, these need to be adjusted manually to get the best results
    alpha = 0.1
    # to count loop
    i = 0
    # loop to be exited as soon as the alpha step bring us away from lowest MSE
    while(current_mse == previous_mse):
        # calculate new parameters with gradient descent algorithm
        a, b = my_datas.gradient_descent(a, b, my_datas.target["y"], alpha)
        # calculate new result
        y_calculated = []
        for x in my_datas.features["x1"]:
            y_calculated.append(my_datas.get_polynomial_expression_result(a, x, [b]))
        # get new MSE
        current_mse = my_datas.MSE(my_datas.target["y"], y_calculated)
        # check if new MSE is smaller than the prveious one
        if current_mse <= previous_mse:
            # if it smaller, update previous mse
            previous_mse = current_mse
        i += 1
    print(f"Took {i} loops to get minimum MSE with alpha={alpha}")
    print(f"Minimum MSE: {current_mse}")
    print(f"Found parameters: a={a}, b={b}")
    print(f"Y calculated with found parameters: {y_calculated}")
    print(f"Y expected: {my_datas.target['y']}")

