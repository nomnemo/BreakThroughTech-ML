import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#  ## Step 1: Define a Function
def f_x(x): 
    result = 0.001 * (3 * x - 1)**4 + 1.5 * (2 * x - 4)**2 + 5 * x + 7
    return result # Do not remove this line of code

# ## Step 1: Find the Minimum of a Function Visually 
xs = np.linspace(-10, 10, 1000)
ys = [f_x(x) for x in xs]
sns.lineplot(x=xs, y=ys) # Do not remove this line of code

# ## Step 2: Find the Minimum of a Function Using a Brute Force Approach
x_pos = np.argmin(ys) 
x_min = xs[x_pos]

# ## Step 3: Find the Minimum of a Function Using Gradient Descent
def gradient(x): 
    
    result = 0.012 * (3 * x - 1)**3 + 12 * x - 19
    return result # Do not remove this line of code

# We are also going to use the 2nd derivative of `f_x`, also known as the "Hessian", to dynamically compute learning rates.
def hessian(x): # Do not remove this line of code
    
    result = 0.108 * (3 * x - 1)**2 + 12
    return result # Do not remove this line of code


def gradient_descent(w_0, #initial starting point (scalar),
                     hessian, # function used to compute learning rate (step size) 
                     gradient, # function used to compute the gradient
                     tolerance=10**-6, #difference for convergence testing
                     max_iter=100 #maximum number of updates to run
                    ):
    
    #record prior value for convergence testing
    w_prior = w_0 
    
    for i in range(max_iter):
        
        #---DO NOT DELETE OR EDIT THE CODE ABOVE THIS LINE-----------------------------|  
        #---Write Your Code Below -----------------------------|
        
        #1. compute gradient at current level using the gradient() function
        grad = gradient(w_prior)
        
        #2. compute learning rate at current level using the hessian() function
        learning_rate = 1/hessian(w_prior)
        
        #3. update the next weight w based on w_prior, learning rate and grad
        w = w_prior - learning_rate*grad
        
        #4. check for convergence
        if np.abs(w-w_prior) < tolerance: 
            break
            
        #5. set w_prior to current w
        w_prior = w 

    
    return w, i


value_of_w, convergence_speed = gradient_descent(0, hessian, gradient)
print(value_of_w)
print(convergence_speed)


# Rather than using the Hessian approach, what if we want to use a constant learning rate?
learning_rate = 0.01

# function that will replaces the hessian() function above
h = lambda x: 1 / learning_rate

gradient_descent(0, h, gradient)


# Notice the difference in convergence speeds between these two methods.
learning_rates = np.linspace(0.01, 0.1, 20)

convergence_speeds = []

for current_learning_rate in learning_rates:
    h = lambda x: 1 / current_learning_rate
    w, convergence_speed = gradient_descent(0, h, gradient)
    convergence_speeds.append(convergence_speed)


# Now let's visualize the results of this analysis. 
fig = plt.figure(figsize=(15,5))
ax = sns.barplot(x=learning_rates, y=convergence_speeds)
g = ax.set_xticklabels([np.round(x, 4) for x in learning_rates])
plt.title('# Iterations to Converge by Learning Rate')
g = ax.set_xlabel('Learning Rate')
g = ax.set_ylabel('Iterations')
g = plt.axhline(y=convergence_speeds[1],color='black')

