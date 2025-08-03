import streamlit as st

st.set_page_config(page_title="🎓 CGPA Calculator", layout="centered")

st.title("🎓 Customizable CGPA Calculator")
st.markdown("Easily calculate your CGPA by entering grades and credit hours. Fully customizable! ✨")

st.sidebar.header("🔧 Customize Grade Points")

# Editable grade-to-point mapping
default_grades = {
    "O": 10,
    "A+": 9,
    "A": 8,
    "B+": 7,
    "B": 6,
    "C": 5,
    "F": 0
}
custom_grade_points = {}

for grade, point in default_grades.items():
    custom_grade_points[grade] = st.sidebar.number_input(f"{grade} Grade Point", min_value=0.0, max_value=10.0, step=0.1, value=float(point))

st.sidebar.markdown("---")
st.sidebar.markdown("🧾 Customize grade point values as per your university")

# Number of subjects
st.markdown("### 📘 Add Subject Details")
num_subjects = st.number_input("Enter number of subjects", min_value=1, max_value=50, step=1)

grades = []
credits = []

with st.form("cgpa_form"):
    for i in range(int(num_subjects)):
        st.markdown(f"**Subject {i+1}**")
        cols = st.columns([1.5, 1.5])
        with cols[0]:
            grade = st.selectbox(
                f"Grade for Subject {i+1}",
                options=list(custom_grade_points.keys()),
                key=f"grade_{i}"
            )
        with cols[1]:
            credit = st.number_input(
                f"Credit for Subject {i+1}",
                min_value=1, max_value=10,
                key=f"credit_{i}"
            )
        grades.append(custom_grade_points[grade])
        credits.append(credit)
        st.markdown("---")

    submitted = st.form_submit_button("📊 Calculate CGPA")

if submitted:
    total_credits = sum(credits)
    total_points = sum([g * c for g, c in zip(grades, credits)])

    if total_credits == 0:
        st.error("❌ Total credits cannot be zero.")
    else:
        cgpa = total_points / total_credits
        st.success(f"🎯 Your CGPA is: **{cgpa:.2f}**")

        # Show detailed breakdown
        with st.expander("📋 Detailed Breakdown"):
            for i in range(int(num_subjects)):
                st.write(f"Subject {i+1}: Grade Point = {grades[i]}, Credit = {credits[i]}, Weighted = {grades[i]*credits[i]}")
            st.markdown(f"**Total Points** = {total_points}")
            st.markdown(f"**Total Credits** = {total_credits}")

        # Optional download as text file
        st.download_button(
            label="📤 Download Report",
            data=f"CGPA Calculation Report\nCGPA: {cgpa:.2f}\nTotal Credits: {total_credits}\nTotal Points: {total_points}",
            file_name="cgpa_report.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("✅ Built with ❤️ using Streamlit | Customize freely | Developed by [Vishal Kumar]")
