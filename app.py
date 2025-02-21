import streamlit as st
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
import os
from streamlit_option_menu import option_menu
import datetime
import streamlit as st
import pandas as pd
from io import StringIO
from sqlalchemy import create_engine, text
import plotly.express as px
import plost
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy.orm import sessionmaker
import datetime

####################################
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
####################################
# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css") 

####################################
# Set your database URI (consider using environment variables for sensitive data)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123@localhost:5432/project")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

####################################
# Sidebar for navigation
####################################
with st.sidebar:
    selected = option_menu(
        'Project',
        ['Employee Performance Insights'],
        menu_icon='hospital-fill',
        icons=['activity'],
        default_index=0
    )

####################################
# Main content   
 ###################################
# ---------------------------------
# Fetch Total Number of Employees
# ---------------------------------
employees_query = "SELECT COUNT(*) AS num_employees FROM employees"
num_employees = pd.read_sql(employees_query, engine).iloc[0]['num_employees']
# ---------------------------------
# Get the Latest Attendance Date
# ---------------------------------
latest_date_query = "SELECT MAX(Date) AS latest_date FROM attendance"
latest_date_result = pd.read_sql(latest_date_query, engine)
latest_date = latest_date_result.iloc[0]['latest_date']

# Ensure we have a valid latest_date and format it nicely
if pd.notnull(latest_date):
    latest_date_str = pd.to_datetime(latest_date).strftime('%B %d, %Y')
    # ---------------------------------
    # Fetch Present Count on Latest Date (assuming present means onLeave = FALSE)
    # ---------------------------------
    present_query = f"""
        SELECT COUNT(*) AS num_present
        FROM attendance
        WHERE date = '{latest_date}' AND onleave = FALSE
    """
    num_present = pd.read_sql(present_query, engine).iloc[0]['num_present']
    # ---------------------------------
    # Fetch On-Leave Count on Latest Date
    # ---------------------------------
    on_leave_query = f"""
        SELECT COUNT(*) AS num_on_leave
        FROM attendance
        WHERE date = '{latest_date}' AND onleave = TRUE
    """
    num_on_leave = pd.read_sql(on_leave_query, engine).iloc[0]['num_on_leave']
else:
    latest_date_str = "N/A"
    num_present = 0
    num_on_leave = 0

# ---------------------------------
# Display Metrics in a 3-Column Layout
# ---------------------------------
st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)

col1.markdown(f'''
<div class="card">
  <div class="metric-title" style="font-weight: bold; font-size: 1.2rem;">Total Number of Employees</div>
  <div class="metric-value" style="font-size: 2rem;">{num_employees}</div>
</div>
''', unsafe_allow_html=True)

col2.markdown(f'''
<div class="card">
  <div class="metric-title" style="font-weight: bold; font-size: 1.2rem;">Present ({latest_date_str})</div>
  <div class="metric-value" style="font-size: 2rem;">{num_present}</div>
</div>
''', unsafe_allow_html=True)

col3.markdown(f'''
<div class="card">
  <div class="metric-title" style="font-weight: bold; font-size: 1.2rem;">On-Leave ({latest_date_str})</div>
  <div class="metric-value" style="font-size: 2rem;">{num_on_leave}</div>
</div><br>
''', unsafe_allow_html=True)

####################################  

########################################################
# Function to fetch existing data
########################################################
import streamlit as st
import pandas as pd
import datetime
from sqlalchemy import create_engine

# Assume you have already set up your engine, for example:
# engine = create_engine("postgresql://postgres:123@localhost:5432/project")

########################################################
# Function to fetch attendance data
########################################################
def fetch_data():
    try:
        query = """
        SELECT 
            A.attendance_id, 
            A.employee_id,
            E.fname, 
            E.lname, 
            A.organization, 
            A.work_setup, 
            A.onleave, 
            A.durationstart, 
            A.durationend, 
            A.leave_type, 
            A.timein, 
            A.timeout, 
            A.date, 
            A.location
        FROM Attendance A
        JOIN Employees E ON A.employee_id = E.employee_id
        ORDER BY A.date DESC;
        """
        df = pd.read_sql(query, con=engine)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

st.markdown("### Attendance Records")

# Fetch attendance data
attendance_data = fetch_data()

if not attendance_data.empty:
    # Create a combined employee name column (keep employee_id for merging)
    attendance_data['employee_name'] = attendance_data['fname'] + ' ' + attendance_data['lname']
    
    # Drop unwanted columns (but keep employee_id)
    attendance_data = attendance_data.drop(columns=['attendance_id', 'fname', 'lname'])
    
    # Convert date and time-related columns to datetime objects
    attendance_data['durationstart'] = pd.to_datetime(attendance_data['durationstart'], errors='coerce')
    attendance_data['durationend']   = pd.to_datetime(attendance_data['durationend'], errors='coerce')
    attendance_data['date']          = pd.to_datetime(attendance_data['date'], errors='coerce')
    attendance_data['timein']        = pd.to_datetime(attendance_data['timein'], format='%H:%M:%S', errors='coerce')
    attendance_data['timeout']       = pd.to_datetime(attendance_data['timeout'], format='%H:%M:%S', errors='coerce')
    
    # Use a date picker with default as the latest attendance date
    selected_date = st.date_input("Filter by Date", value=attendance_data['date'].max().date())
    
    # Filter attendance records for the selected date
    present_data = attendance_data[attendance_data['date'].dt.date == selected_date].copy()
    
    # Compute the "Note" for present records
    def compute_note(row):
        # If onLeave is True and the leave period covers the selected date, mark as "OnLeave"
        if row['onleave'] and pd.notnull(row['durationstart']) and pd.notnull(row['durationend']):
            if row['durationstart'].date() <= selected_date <= row['durationend'].date():
                return "OnLeave"
        # Otherwise, check timein
        if pd.isna(row['timein']):
            return "Late"
        elif row['timein'].time() <= datetime.time(8, 0):
            return "On-time"
        else:
            return "Late"
    
    present_data['Note'] = present_data.apply(compute_note, axis=1)
    
    # Format columns for present records
    present_data['durationstart'] = present_data['durationstart'].dt.strftime('%B %d, %Y %I:%M %p')
    present_data['durationend']   = present_data['durationend'].dt.strftime('%B %d, %Y %I:%M %p')
    present_data['timein']        = present_data['timein'].dt.strftime('%I:%M %p')
    present_data['timeout']       = present_data['timeout'].dt.strftime('%I:%M %p')
    present_data['date']          = present_data['date'].dt.strftime('%B %d, %Y')
    
    # -----------------------------------
    # Identify Absent Employees
    # -----------------------------------
    # Fetch all employees from the Employees table
    employee_query = "SELECT employee_id, fname, lname, (fname || ' ' || lname) AS employee_name FROM Employees"
    employee_data = pd.read_sql(employee_query, engine)
    
    # Identify employees with no attendance record for the selected date
    present_ids = set(present_data['employee_id'])
    absent_employees = employee_data[~employee_data['employee_id'].isin(present_ids)].copy()
    
    # Create a DataFrame for absent employees with columns matching the final output
    absent_data = pd.DataFrame({
         'employee_id': absent_employees['employee_id'],
         'employee_name': absent_employees['employee_name'],
         'organization': "", 
         'work_setup': "",
         'onleave': False,
         'leave_type': "",
         'durationstart': "",
         'durationend': "",
         'timein': "",
         'timeout': "",
         'date': selected_date.strftime('%B %d, %Y'),
         'location': "",
         'Note': "Absent"
    })
    
    # Combine present and absent data
    combined_data = pd.concat([present_data, absent_data], ignore_index=True)
    
    # Reorder columns as desired
    cols = ['employee_name', 'organization', 'work_setup', 'onleave', 'leave_type',
            'durationstart', 'durationend', 'timein', 'timeout', 'date', 'location', 'Note']
    combined_data = combined_data[cols]
    
    # Optional: Sort by employee name
    combined_data = combined_data.sort_values(by='employee_name')
    
    st.dataframe(combined_data)
else:
    st.warning("No attendance records found.")


#########################################
# Process attendance_data for a detailed line chart
#########################################

# Convert 'date' column to datetime for proper grouping
attendance_data['date'] = pd.to_datetime(attendance_data['date'], errors='coerce')

# Define a function to classify attendance
def classify_attendance(row):
    # If employee is on leave, classify as On-Leave and pass leave details
    if row['onleave']:
        return 'On-Leave', row['leave_type'], row['durationstart'], row['durationend']
    # If timein is missing (and not on leave), classify as Absent
    if pd.isna(row['timein']):
        return 'Absent', None, None, None
    # Attempt to parse timein (assumed format 'HH:MM:SS')
    try:
        timein_dt = pd.to_datetime(row['timein'], format='%H:%M:%S', errors='coerce')
        if pd.isna(timein_dt):
            return 'Unknown', None, None, None
        timein = timein_dt.time()
    except Exception as e:
        st.warning(f"Error converting timein: {e}")
        return 'Unknown', None, None, None
    # Check if employee is Late (arriving at or after 8:00 AM)
    cutoff = datetime.time(8, 0, 0)
    if timein >= cutoff:
        return 'Late', None, None, None
    # Otherwise, the employee is Present
    return 'Present', None, None, None

# Apply classification function to each row, creating new columns
attendance_data[['Status', 'Leave Type', 'Leave Start', 'Leave End']] = attendance_data.apply(
    lambda row: pd.Series(classify_attendance(row)), axis=1
)

# -----------------------------
# Aggregation for Union (Present includes Late)
# -----------------------------
# Create a new column 'Attendance' that groups both "Present" and "Late" as "Present"
attendance_data['Attendance'] = attendance_data['Status'].apply(
    lambda x: 'Present' if x in ['Present', 'Late'] else x
)

# Aggregate daily counts based on the union grouping
attendance_union = attendance_data.groupby(['date', 'Attendance']).size().reset_index(name='Count')
total_counts = attendance_data.groupby('date').size().reset_index(name='Total Employees')
attendance_union = pd.merge(attendance_union, total_counts, on='date')
attendance_union['Percentage'] = (attendance_union['Count'] / attendance_union['Total Employees']) * 100

# Pivot table for union data – we'll have "Present" (union) and "On-Leave"
union_pivot = attendance_union.pivot_table(
    index='date', columns='Attendance', values='Percentage', aggfunc='sum'
).reset_index()
# Ensure both keys exist
for key in ['Present', 'On-Leave']:
    if key not in union_pivot.columns:
        union_pivot[key] = 0

# -----------------------------
# Aggregation for Late only
# -----------------------------
late_data = attendance_data[attendance_data['Status'] == 'Late']
late_counts = late_data.groupby('date').size().reset_index(name='Late_Count')
late_counts = pd.merge(late_counts, total_counts, on='date')
late_counts['Late_Percentage'] = (late_counts['Late_Count'] / late_counts['Total Employees']) * 100

# -----------------------------
# Merge union and late data
# -----------------------------
status_pivot = pd.merge(union_pivot, late_counts[['date', 'Late_Percentage']], on='date', how='left')
status_pivot['Late_Percentage'] = status_pivot['Late_Percentage'].fillna(0)

# Rename for clarity
status_pivot = status_pivot.rename(columns={'Late_Percentage': 'Late'})

# Ensure our pivot table has the desired columns and fill missing with 0
expected_statuses = ['Present', 'Late', 'On-Leave']
status_pivot = status_pivot[['date'] + expected_statuses]
for col in expected_statuses:
    status_pivot[col] = status_pivot[col].fillna(0)

####################################################################################
# ROW 3 
####################################################################################
# Create the line chart using Plotly Express
# -----------------------------
fig_line = px.line(
    status_pivot, 
    x='date', 
    y=expected_statuses, 
    title='Daily Attendance Summary',
    labels={'value': 'Percentage', 'date': 'Date'},
    markers=True
)
fig_line.update_layout(
    xaxis=dict(tickformat="%B %d, %Y")
)

# Customize the hover information for each trace
for trace in fig_line.data:
    if trace.name == 'Present':
        trace.hovertemplate = (
            'Date: %{x|%B %d, %Y}<br>'
            'Present (incl. Late): %{y:.2f}%<br>'
            '<extra></extra>'
        )
    elif trace.name == 'Late':
        trace.hovertemplate = (
            'Date: %{x|%B %d, %Y}<br>'
            'Late: %{y:.2f}%<br>'
            '<extra></extra>'
        )
    elif trace.name == 'On-Leave':
        trace.hovertemplate = (
            'Date: %{x|%B %d, %Y}<br>'
            'On-Leave: %{y:.2f}%<br>'
            '<extra></extra>'
        )

st.plotly_chart(fig_line)
########################################################
# Load task data for bar chart
########################################################
def load_task_data():
    query = "SELECT state, COUNT(*) as count FROM Story_points GROUP BY state ORDER BY count DESC"
    return pd.read_sql(query, engine)

task_data = load_task_data()

#-------------------------------------------------------
# Load leave data for the current week for pie chart
#-------------------------------------------------------
query_leave = """
SELECT leave_type, COUNT(*) as count FROM Attendance 
WHERE leave_type IS NOT NULL GROUP BY leave_type;
"""
leave_data = pd.read_sql(query_leave, engine)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Task Status Breakdown")
    fig_bar = px.bar(task_data, x='state', y='count',
                     labels={'state': 'Task State', 'count': 'Number of Tasks'},
                     height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### Leave Breakdown")
    if leave_data.empty:
        st.info("No leave records for the current week.")
    else:
        # Create the pie chart
        fig_pie = px.pie(leave_data, names='leave_type', values='count')

        # Update layout to adjust size and position the legend below the chart
        fig_pie.update_layout(
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",
                y=-0.2,  #move the legend up or down
                xanchor="center",
                x=0.5  # Center the legend
            ),
            height=400,  # Adjust the height
            width=600    # Adjust the width
        )

        # Display the pie chart in Streamlit
        st.plotly_chart(fig_pie, use_container_width=True)

########################################################
# -------------------------------
# 🔹 Database Connection
# -------------------------------
st.subheader("Import Data")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    new_data = pd.read_csv(stringio)

    # Display uploaded data
    st.write("📌 **Preview of uploaded data:**")
    st.dataframe(new_data)

    # -------------------------------------------
    # 1️⃣ Attendance Table Import - highest priority
    #    Check for attendance-specific columns.
    # -------------------------------------------
    if all(col in new_data.columns for col in ["employee_id", "date", "organization", "work_setup"]):
        if st.button("Import Attendance to Database"):
            try:
                # Check if Attendance table exists
                check_table_query = (
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='attendance');"
                )
                table_exists = session.execute(text(check_table_query)).scalar()

                if not table_exists:
                    st.error("⚠️ Error: 'attendance' table does not exist in the database.")
                else:
                    # Get existing attendance records to ensure one record per employee per day
                    existing_attendance = pd.read_sql(
                        "SELECT employee_id, date FROM attendance;", con=engine
                    )

                    # Merge to filter out duplicates (i.e., same employee_id and date)
                    new_attendance = new_data.merge(
                        existing_attendance, on=["employee_id", "date"], how="left", indicator=True
                    )
                    new_attendance = new_attendance[new_attendance["_merge"] == "left_only"].drop(
                        columns=["_merge"]
                    )

                    if not new_attendance.empty:
                        new_attendance.to_sql(
                            "attendance", con=engine, if_exists="append", index=False, method="multi"
                        )
                        session.commit()
                        st.success("✅ Attendance data imported successfully!")
                    else:
                        st.warning("⚠️ Data Already Exist.")
            except Exception as e:
                session.rollback()
                st.error(f"❌ Error importing attendance data: {e}")

    # -------------------------------------------
    # 2️⃣ Story Points Table Import
    #    Check for employee_id and task columns.
    # -------------------------------------------
    elif all(col in new_data.columns for col in ["employee_id", "task"]) and "organization" not in new_data.columns:
        if st.button("Import Story Points to Database"):
            try:
                # Check if Story Points table exists
                check_table_query = (
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='story_points');"
                )
                table_exists = session.execute(text(check_table_query)).scalar()

                if not table_exists:
                    st.error("⚠️ Error: 'story_points' table does not exist in the database.")
                else:
                    # Verify that all employee IDs exist in the Employees table
                    existing_employees = pd.read_sql("SELECT employee_id FROM employees;", con=engine)
                    missing_employees = set(new_data["employee_id"]) - set(existing_employees["employee_id"])

                    if missing_employees:
                        st.error(f"⚠️ Error: The following employee IDs do not exist: {missing_employees}")
                    else:
                        # Get existing story points using employee_id and task as unique identifiers
                        existing_story_points = pd.read_sql("SELECT employee_id, task FROM story_points;", con=engine)

                        # Merge to filter out duplicates based on employee_id and task
                        new_story_points = new_data.merge(
                            existing_story_points, on=["employee_id", "task"], how="left", indicator=True
                        )
                        new_story_points = new_story_points[new_story_points["_merge"] == "left_only"].drop(
                            columns=["_merge"]
                        )

                        if not new_story_points.empty:
                            new_story_points.to_sql(
                                "story_points", con=engine, if_exists="append", index=False, method="multi"
                            )
                            session.commit()
                            st.success("✅ Story Points data imported successfully!")
                        else:
                            st.warning("⚠️ Story points data already exist.")
            except Exception as e:
                session.rollback()
                st.error(f"❌ Error importing Story Points data: {e}")

    # -------------------------------------------
    # 3️⃣ Employees Table Import - lowest priority
    #    Check for fname and lname columns.
    # -------------------------------------------
    elif all(col in new_data.columns for col in ["fname", "lname"]):
        if st.button("Import Employees to Database"):
            try:
                # Check if Employees table exists
                check_table_query = (
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='employees');"
                )
                table_exists = session.execute(text(check_table_query)).scalar()

                if not table_exists:
                    st.error("⚠️ Error: 'employees' table does not exist in the database.")
                else:
                    # Get existing employees
                    existing_employees = pd.read_sql("SELECT fname, lname FROM employees;", con=engine)

                    # Merge and filter out duplicates
                    new_employees = new_data.merge(
                        existing_employees, on=["fname", "lname"], how="left", indicator=True
                    )
                    new_employees = new_employees[new_employees["_merge"] == "left_only"].drop(
                        columns=["_merge"]
                    )

                    if not new_employees.empty:
                        new_employees[["fname", "lname"]].to_sql(
                            "employees", con=engine, if_exists="append", index=False, method="multi"
                        )
                        session.commit()
                        st.success("✅ Employee data imported successfully!")
                    else:
                        st.warning("⚠️ Eemployees already exist.")
            except Exception as e:
                session.rollback()
                st.error(f"❌ Error importing employee data: {e}")

    else:
        st.error("⚠️ Invalid CSV format! Ensure it contains Employees, Attendance, or Story Points data.")

# Close the session properly
session.close()
