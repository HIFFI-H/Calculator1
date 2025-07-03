import streamlit as st

# Initialize session state
if "expression" not in st.session_state:
    st.session_state.expression = ""

# Functions to handle button actions
def append_char(char):
    st.session_state.expression += str(char)

def calculate():
    try:
        st.session_state.expression = str(eval(st.session_state.expression))
    except:
        st.session_state.expression = "Error"

def clear():
    st.session_state.expression = ""

# Calculator Layout
st.markdown("<h2 style='text-align: center;'> HiFFi Calculator</h2>", unsafe_allow_html=True)
st.write("")  # Spacer
st.markdown("----------")
buttons = [
    ['7', '8', '9', '+'],
    ['4', '5', '6', '-'],
    ['1', '2', '3', '*'],
    ['0', 'C', '=', '/']
]

# Display buttons in rows
for row in buttons:
    cols = st.columns(4)
    for i, btn in enumerate(row):
        with cols[i]:
            if st.button(btn, use_container_width=True):
                if btn == "C":
                    clear()
                else:
                    append_char(btn)

# Expression display
st.markdown("###")
st.text_input("Expression", value=st.session_state.expression, label_visibility="collapsed", disabled=True)

# Calculate button
if st.button("Calculate"):
    calculate()

st.markdown("----------")
st.markdown("<h2 style='text-align: center;'> Thanks for Using</h2>", unsafe_allow_html=True)

st.subheader("Made by:blue[Irfan Ali] :sunglasses:")