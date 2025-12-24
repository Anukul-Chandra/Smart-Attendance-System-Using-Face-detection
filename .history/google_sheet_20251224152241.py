import gspread
from google.oauth2.service_account import Credentials

def get_sheet():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(
        "credentials.json",
        scopes=scopes
    )
    client = gspread.authorize(creds)
    sheet = client.open("Smart Attendance").sheet1
    return sheet

def mark_attendance_sheet(name, date, time, status="Present"):
    sheet = get_sheet()
    sheet.append_row([name, date, time, status])
