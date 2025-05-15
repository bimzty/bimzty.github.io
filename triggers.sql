CREATE OR REPLACE TRIGGER sale_del_cascade
BEFORE DELETE ON Sale
FOR EACH ROW

BEGIN
    DELETE FROM Construction_Project
    WHERE Construction_Project.project_id = :old.cp_project_id;

    DELETE FROM Escrow
    WHERE Escrow.escrow_id = :old.escrow_escrow_id;

    DELETE FROM Chosen_Style
    WHERE Chosen_Style.sale_invoice_id = :old.invoice_id;

END;

/
CREATE OR REPLACE TRIGGER cp_del_cascade
BEFORE DELETE ON Construction_Project
FOR EACH ROW

BEGIN
    DELETE FROM Task
    WHERE Task.cp_project_id = :old.project_id;

    DELETE FROM Decorator_Choice
    WHERE Decorator_Choice.cp_project_id = :old.project_id;

END;

/
CREATE OR REPLACE TRIGGER house_style_del_cascade
AFTER DELETE ON House_Style
FOR EACH ROW

BEGIN
    DELETE FROM Elevation
    WHERE Elevation.house_style_style_name = :old.style_name;

    DELETE FROM Room
    WHERE Room.house_style_style_name = :old.style_name;
END;

/
-- DROP TRIGGER update_style_size;
-- DROP TRIGGER update_lot_premium_by_elevation_cost;
-- DROP TRIGGER update_lot_premium_by_opt_price;
-- DROP TRIGGER update_project_pct_complete;

CREATE OR REPLACE TRIGGER update_style_size
AFTER INSERT OR UPDATE
ON Room
DECLARE
    CURSOR style_cursor IS
        SELECT house_style_style_name
        FROM Room
        GROUP BY house_style_style_name;

    v_total_size House_Style.style_size%TYPE := 0;
BEGIN
    FOR style_rec IN style_cursor LOOP
        SELECT SUM(r.room_size) INTO v_total_size
        FROM Room r
        WHERE r.house_style_style_name = style_rec.house_style_style_name;

        UPDATE House_Style
        SET style_size = v_total_size
        WHERE style_name = style_rec.house_style_style_name;

        -- DBMS_OUTPUT.PUT_LINE('Style size of House Style <' || style_rec.house_style_style_name || '> is updated to ' || v_total_size);
    END LOOP;
END;

/
CREATE OR REPLACE TRIGGER update_lot_premium_by_elevation_cost
AFTER INSERT OR UPDATE
ON Chosen_Style
FOR EACH ROW
DECLARE
    v_elevation_cost    Elevation.elevation_cost%TYPE := 0;
    v_lot_premium       Sale.lot_premium%TYPE := 0;

BEGIN
    SELECT s.lot_premium INTO v_lot_premium
    FROM Sale s
    WHERE s.invoice_id = :new.sale_invoice_id;

    SELECT e.elevation_cost INTO v_elevation_cost
    FROM Elevation e
    WHERE e.elevation_id = :new.elevation_chosen
    AND e.house_style_style_name = :new.house_style_style_name;

    UPDATE Sale
    SET lot_premium = v_lot_premium + v_elevation_cost
    WHERE invoice_id = :new.sale_invoice_id;

    -- DBMS_OUTPUT.PUT_LINE('Lot premium of Sale with invoice_id <' ||
    -- :new.sale_invoice_id || '> is updated to ' || (v_lot_premium + v_elevation_cost));

END;

/
CREATE OR REPLACE TRIGGER update_project_pct_complete
AFTER INSERT OR UPDATE
ON Task
DECLARE
    v_task_pct_complete     Task.task_pct_complete%TYPE := 0;
    v_task_count            NUMBER := 0;
    v_task_completed        NUMBER := 0;

    CURSOR project_cursor IS
        SELECT cp_project_id
        FROM Task
        GROUP BY cp_project_id;

BEGIN
    FOR project_rec IN project_cursor LOOP
        v_task_count := 0;
        v_task_completed := 0;

        FOR task_rec IN (SELECT task_pct_complete
                         FROM Task
                         WHERE cp_project_id = project_rec.cp_project_id) LOOP
            v_task_count := v_task_count + 1;
            IF task_rec.task_pct_complete = 100 THEN
                v_task_completed := v_task_completed + 1;
            END IF;
        END LOOP;

        IF v_task_count > 0 THEN
            UPDATE Construction_Project
            SET project_pct_complete = (v_task_completed / v_task_count) * 100
            WHERE project_id = project_rec.cp_project_id;

            -- DBMS_OUTPUT.PUT_LINE('Project percentage completed of Construction Project with project id <' ||
            --                      project_rec.cp_project_id || '> is updated to ' ||
            --                      ROUND((v_task_completed / v_task_count) * 100, 2) || '%');
        ELSE
            DBMS_OUTPUT.PUT_LINE('No tasks found for project id <' ||
                                 project_rec.cp_project_id || '>.');
        END IF;
    END LOOP;
END;

-- /
-- -- DROP TRIGGER void_sale
-- -- DROP TRIGGER delete_sale
--
-- CREATE OR REPLACE TRIGGER void_sale
-- BEFORE UPDATE OF contract_status
-- ON Sale
-- FOR EACH ROW
-- DECLARE
--     v_refund NUMBER(7,2) := 0;
-- BEGIN
--     IF :new.contract_status = 'Void' THEN
--
--         DELETE FROM Chosen_Style
--         WHERE sale_invoice_id = :old.invoice_id;
--
--         -- DELETE FROM Task
--         -- WHERE Task.cp_project_id = :old.cp_project_id;
--         --
--         -- DELETE FROM Decorator_Choice
--         -- WHERE Decorator_Choice.cp_project_id = :old.cp_project_id;
--
--         DELETE FROM Construction_Project
--         WHERE project_id = :old.cp_project_id;
--
--         DELETE FROM Escrow
--         WHERE escrow_id = :old.escrow_escrow_id;
--
--         v_refund := calc_refund(:old.invoice_id);
--         DBMS_OUTPUT.PUT_LINE('Sale record is voided and the refund is: ' || v_refund);
--     END IF;
-- END;
--
-- /
-- CREATE OR REPLACE TRIGGER delete_sale
-- BEFORE DELETE ON Sale
-- FOR EACH ROW
--
-- BEGIN
--     RAISE_APPLICATION_ERROR(-20001, 'Deletion of Sale is not allowed. Sale status should be set to Void instead.');
-- END;
-- /
