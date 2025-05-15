SET SERVEROUTPUT ON;
-- ========================== Procedure: Filter Options By Category ==========================
CREATE OR REPLACE PROCEDURE FilterOptionsByCategory(
    p_Category IN VARCHAR2
)
AS
BEGIN
    FOR rec IN (
        SELECT option_id, option_desc, option_cost
        FROM option_list
        WHERE (p_Category IS NULL OR option_category = p_Category)
        ORDER BY option_id
    )
    LOOP
        DBMS_OUTPUT.PUT_LINE('Option ID: ' || rec.option_id ||
                             ', Description: ' || rec.option_desc ||
                             ', Cost: ' || rec.option_cost);
    END LOOP;
END FilterOptionsByCategory;
/

-- ========================== Procedure: Filter Options By Category ==========================


-- ========================== Procedure: Create sales record ==========================
-- CREATE OR REPLACE PROCEDURE generate_next_id (
--     p_table_name IN VARCHAR2,
--     p_column_name IN VARCHAR2,
--     p_next_id OUT NUMBER
-- )
-- AS
-- BEGIN
--     EXECUTE IMMEDIATE
--         'SELECT NVL(MAX(' || p_column_name || '), 0) + 1 FROM ' || p_table_name
--     INTO p_next_id;
-- END generate_next_id;
--
-- /
--
-- CREATE OR REPLACE PROCEDURE create_sale_record(
--     p_customer_id customer.customer_id%TYPE,
--     p_sr_liscense_no sales_representative.liscense_no%TYPE,
--     p_bank_id bank.bank_id%TYPE,
--     p_lot_id sale.lot_lot_id%TYPE,
--     p_financing_method sale.financing_method%TYPE,
--     p_e_agent_id escrow.escrow_agent_id%TYPE,
--     p_e_deposit_amt escrow.deposit_amount%TYPE,
--     p_reversed_style chosen_style.reversed_style%TYPE,
--     p_elevation_chosen chosen_style.elevation_chosen%TYPE,
--     p_house_style_style_name chosen_style.house_style_style_name%TYPE,
--     p_cm_employee_id construction_manager.employee_id%TYPE
-- )
-- AS
--     v_escrow_id escrow.escrow_id%TYPE;
--     v_schoice_id chosen_style.schoice_id%TYPE;
--     v_project_id construction_project.project_id%TYPE;
--     v_invoice_id sale.invoice_id%TYPE;
-- BEGIN
--     -- Generate IDs
--     generate_next_id('escrow', 'escrow_id', v_escrow_id);
--     generate_next_id('chosen_style', 'schoice_id', v_schoice_id);
--     generate_next_id('construction_project', 'project_id', v_project_id);
--     generate_next_id('sale', 'invoice_id', v_invoice_id);
--
--     -- Create a blank escrow
--     INSERT INTO escrow (escrow_id, escrow_agent_id, deposit_amount)
--     VALUES (v_escrow_id, p_e_agent_id, p_e_deposit_amt);
--
--     -- Create construction_project
--     INSERT INTO construction_project(project_id, start_date, current_stage, project_pct_complete, cm_employee_id)
--     VALUES (v_project_id, SYSDATE, 1, 0, p_cm_employee_id);
--
--     -- CREATE SALE
--     INSERT INTO sale(invoice_id, date_sold, financing_method, lot_lot_id, escrow_escrow_id, cp_project_id, sr_liscense_no, customer_customer_id, bank_bank_id)
--     VALUES (v_invoice_id, SYSDATE, p_financing_method, p_lot_id, v_escrow_id, v_project_id, p_sr_liscense_no, p_customer_id, p_bank_id);
--
--     -- Create chosen_style
--     INSERT INTO chosen_style(schoice_id, reversed_style, elevation_chosen, house_style_style_name, sale_invoice_id)
--     VALUES (v_schoice_id, p_reversed_style, p_elevation_chosen, p_house_style_style_name, v_invoice_id);
--
--     -- Create a dbms_output here
--     dbms_output.put_line('Sale record ' || v_invoice_id || ' created, Escrow ID: ' || v_escrow_id || ' Construction Project ID: ' || v_project_id);
--     COMMIT;
-- EXCEPTION
--     WHEN OTHERS THEN
--         ROLLBACK;
--         RAISE;
-- END create_sale_record;
/
-- ========================== Procedure: Create sales record ==========================
