REM This report is for monitoring the status of ongoing construction projects, including project details, associated tasks, and their completion percentages.
REM This report helps Eggshell Home Builders understand project progress and identify any potential delays or areas requiring attention

SELECT 
    cp.project_id, 
    cp.current_stage, 
    cp.project_pct_complete AS project_completion,
    cm.crew_no AS manager_id, 
    e.employee_fname || ' ' || e.employee_lname AS manager_name, 
    dc.dchoice_id,
    dc.choice_selected_date as choice_date,
    dc.item_quantity,
    t.task_id,
    t.task_desc AS TASK_DESCRIPTION, 
    t.task_pct_complete AS task_completion    
FROM Decorator_Choice dc
JOin Construction_Project cp ON dc.cp_project_id = cp.project_id
JOIN 
    Task t ON cp.project_id = t.cp_project_id
JOIN 
    Construction_Manager cm ON cm.employee_id = cm.crew_no
JOIN Employee e ON e.employee_id = cm.employee_id
ORDER BY 
    cp.project_id, t.task_id;
