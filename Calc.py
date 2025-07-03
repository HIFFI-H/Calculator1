import streamlit as st

def main ():

    st.title("HiFFi Calculator")
    st.markdown("------")
# Input values

    num1 = st.number_input("Enter first number", value = 0.0, step = 0.1 )
    num2 = st.number_input("Enter second number", value = 0.0, step = 0.1 )

    st.markdown("------")

    operation = st.selectbox(
        "Select Operation",
        ("Add", "Sub", "Multi", "Div")
    )
    result = None

    if st.button("Calculate"):
        if operation == "Add":
            result = num1 + num2
        elif operation == "Sub":
            result = num1 - num2
        elif operation == "Multi":
            result = num1 * num2
        elif operation == "Div":
            if num2 != 0:
                result = num1 / num2
            else:
                st.error("Error: Division by zero is not allowed.")
                result = "Undefined"
        if result is not None:
            st.success(f"The result is: {result}")

    st.subheader("Thanks for Using")
    st.markdown("------")


if __name__ == "__main__":
    main()

