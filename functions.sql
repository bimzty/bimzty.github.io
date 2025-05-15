CREATE OR REPLACE FUNCTION get_next_elevation_id(p_house_style VARCHAR2)
RETURN NUMBER
IS
  v_max_id NUMBER;
BEGIN
  SELECT NVL(MAX(elevation_id), -1)
  INTO v_max_id
  FROM Elevation
  WHERE house_style_style_name = p_house_style;

  IF v_max_id = -1 THEN
    RETURN 0;
  ELSE
    RETURN v_max_id + 1;
  END IF;
END;
/

CREATE OR REPLACE FUNCTION get_next_room_id(p_house_style VARCHAR2)
RETURN NUMBER
IS
  v_max_id NUMBER;
BEGIN
  SELECT NVL(MAX(room_id), -1)
  INTO v_max_id
  FROM Room
  WHERE house_style_style_name = p_house_style;

  IF v_max_id = -1 THEN
    RETURN 0;
  ELSE
    RETURN v_max_id + 1;
  END IF;
END;
/

CREATE OR REPLACE FUNCTION get_next_task_id(v_cp_project_id Task.cp_project_id%TYPE)
RETURN NUMBER
IS
  v_max_id NUMBER;
BEGIN
  SELECT NVL(MAX(task_id), -1)
  INTO v_max_id
  FROM Task
  WHERE cp_project_id = v_cp_project_id;

  IF v_max_id = -1 THEN
    RETURN 0;
  ELSE
    RETURN v_max_id + 1;
  END IF;
END;
/

-- CREATE OR REPLACE FUNCTION calc_refund (v_invoice_id IN Sale.invoice_id%TYPE)
-- RETURN NUMBER
-- AS
--     v_deposit   Escrow.deposit_amount%TYPE;
--     v_date_sold Sale.date_sold%TYPE;
--     v_proj_pct  Construction_Project.project_pct_complete%TYPE;
--     v_refund    NUMBER(7,2) := 0;
--
-- BEGIN
--     SELECT e.deposit_amount, s.date_sold, cp.project_pct_complete
--     INTO v_deposit, v_date_sold, v_proj_pct
--     FROM Escrow e, Sale s, Construction_Project cp
--     WHERE v_invoice_id = s.invoice_id
--     AND e.escrow_id = s.escrow_escrow_id
--     AND cp.project_id = s.cp_project_id;
--
--     IF MONTHS_BETWEEN(SYSDATE, v_date_sold) <= 12 THEN
--         v_refund := v_deposit * v_proj_pct / 100;
--         RETURN v_refund;
--     ELSE
--         RETURN 0;
--     END IF;
--
-- EXCEPTION
--     WHEN NO_DATA_FOUND THEN
--         DBMS_OUTPUT.PUT_LINE('No data found for invoice_id: ' || v_invoice_id);
--         return 0;
--     WHEN OTHERS THEN
--         DBMS_OUTPUT.PUT_LINE('Error: ' || SQLERRM);
--         RETURN 0;
--
-- END;
