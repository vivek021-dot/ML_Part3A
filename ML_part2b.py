import pickle

# w=read, b=binary
with open("my_ml_model1.ml", 'rb') as f:
    classifier = pickle.load(f)

while True:
    var1 = float(input("Enter value for account length:"))
    var2 = float(input("Enter value for number vmail messages:"))
    pred = classifier.predict([[var1, var2]])
    print(pred)
    if pred == [1]:
        print("Yes, likely will churn.")
    else:
        print("Unlikely will churn.")
    cont = input("Do you want to continue? y/n")
    if cont == "n":
        break