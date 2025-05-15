REM This report is to provide insights into the performance of home sales
REM This report helps Eggshell Home Builders track which styles and lots are most popular, and the performance of sales representatives.

SELECT 
    s.invoice_id, 
    sr.liscense_no AS sales_rep_id, 
    e.employee_fname || ' ' || e.employee_lname AS sales_rep_name, 
    hs.style_name,
    cs.elevation_chosen,
    hs.base_price,
    s.lot_premium, 
    s.date_sold
FROM 
    Sale s
JOIN 
    Sales_Representative sr ON s.sr_liscense_no  = sr.liscense_no
JOIN
    Employee e ON e.employee_id = sr.employee_id
JOIN 
    chosen_style cs on s.invoice_id = cs.sale_invoice_id
JOIN
    house_style hs on hs.style_name = cs.house_style_style_name
ORDER BY 
    s.date_sold DESC;

