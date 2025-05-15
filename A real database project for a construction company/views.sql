-- ========================== View: Project Stage for Clients==========================
CREATE OR REPLACE VIEW ACCESSIBLE_PROJECTS AS
SELECT
    s.customer_customer_id AS CUSTOMER_ID,
    c.project_id as PROJECT_ID,
    cs.house_style_style_name AS HOUSE_STYLE,
    cs.elevation_chosen AS ELEVATION,
    s.date_sold AS SALE_DATE,
    c.start_date AS CONSTRUCTION_START_DATE,
    c.est_completion_date AS ESTIMATED_COMPLETION_DATE,
    c.project_pct_complete AS COMPLETION_PERCENTAGE,
    CASE
        WHEN c.current_stage BETWEEN 1 AND 3 THEN 1
        WHEN c.current_stage BETWEEN 4 AND 6 THEN 4
        WHEN c.current_stage = 7 THEN 7
    END AS PROJECT_STAGE
FROM SALE s
JOIN construction_project c ON s.cp_project_id = c.project_id
JOIN chosen_style cs ON s.invoice_id = cs.sale_invoice_id
WHERE c.current_stage IN (1, 2, 3, 4, 5, 6, 7) AND
s.CUSTOMER_CUSTOMER_ID = (SELECT customer_id FROM Customer WHERE customer.customer_lname = USER);


-- ========================== View: Options Availabile==========================
/*
SELECT s.invoice_id,cp.current_stage, ol.option_id, ol.stage_no, ol.style_name, ol.option_cost, ol.option_category, ol.option_desc
FROM sale s
JOIN construction_project cp ON s.cp_project_id = cp.project_id
JOIN option_list ol ON cp.current_stage <= ol.stage_no
WHERE cp.current_stage IN (1, 2, 3, 4, 5, 6, 7) AND
 invoice_id = 4 AND contract_status = 'Valid';
 */

-- DROP MATERIALIZED VIEW PROJECT_OPTIONS_VIEW;

CREATE OR REPLACE VIEW PROJECT_OPTIONS_VIEW AS
SELECT
    ap.CUSTOMER_ID,
    ap.PROJECT_ID,
    --ap.HOUSE_STYLE,
    --ap.ELEVATION,
    ap.SALE_DATE,
    --ap.CONSTRUCTION_START_DATE,
    --ap.ESTIMATED_COMPLETION_DATE,
    ap.PROJECT_STAGE as CURRENT_STAGE,
    ol.option_id AS OPTION_ID,
    ol.style_name AS OPTION_NAME,
    ol.stage_no AS SELECT_BY_STAGE,
    ol.option_cost AS OPTION_PRICE
FROM ACCESSIBLE_PROJECTS ap
JOIN option_list ol ON ap.PROJECT_STAGE >= ol.stage_no
WHERE ap.CUSTOMER_ID = (SELECT customer_id FROM Customer WHERE customer.customer_lname = USER);

-- ========================== View: CustomerSaleRecord ==========================

--Create view for restricted information to that particular customer
-- view for sale
CREATE OR REPLACE VIEW CustomerSaleRecord AS
SELECT
    s.invoice_id,
    s.lot_premium,
    s.date_sold,
    s.financing_method
FROM
    Sale s
WHERE
    s.CUSTOMER_CUSTOMER_ID = (SELECT customer_id FROM Customer WHERE customer.customer_lname = USER);
