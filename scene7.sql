-- Insert test data. These 4 customers should be removed.
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'Sarah', 'Johnson', '303 Oak St', 'Phoenix', '85001');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'David', 'Lee', '404 Cedar St', 'San Antonio', '78201');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'Laura', 'Martinez', '505 Maple St', 'San Diego', '92101');
INSERT INTO Customer VALUES (customer_id_seq.NEXTVAL, 'James', 'Anderson', '606 Birch St', 'Dallas', '75201');

-- Display initial data
SELECT * FROM CUSTOMER;
SELECT * FROM SALE;

-- Ensure the job exists and is enabled
BEGIN
    DBMS_SCHEDULER.ENABLE('YEARLY_INACTIVE_CUSTOMER_CLEANUP');
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -27366 THEN  -- ORA-27366: job "..." is already enabled
            RAISE;
        END IF;
END;
/

-- Set the start date to now and run the job immediately
DECLARE
    v_job_exists NUMBER;
BEGIN
    -- Check if the job is already running
    SELECT COUNT(*)
    INTO v_job_exists
    FROM user_scheduler_running_jobs
    WHERE job_name = 'YEARLY_INACTIVE_CUSTOMER_CLEANUP';

    IF v_job_exists = 0 THEN
        -- Set the start date to now
        DBMS_SCHEDULER.SET_ATTRIBUTE(
            name      => 'YEARLY_INACTIVE_CUSTOMER_CLEANUP',
            attribute => 'START_DATE',
            value     => SYSTIMESTAMP
        );

        -- Run the job immediately
        DBMS_SCHEDULER.RUN_JOB('YEARLY_INACTIVE_CUSTOMER_CLEANUP');
    ELSE
        DBMS_OUTPUT.PUT_LINE('Job is already running. Skipping execution.');
    END IF;
END;
/

-- Wait for the job to complete
DECLARE
    v_job_state VARCHAR2(30);
BEGIN
    LOOP
        SELECT state
        INTO v_job_state
        FROM user_scheduler_jobs
        WHERE job_name = 'YEARLY_INACTIVE_CUSTOMER_CLEANUP';

        EXIT WHEN v_job_state = 'SCHEDULED';
        DBMS_SESSION.SLEEP(1); -- Wait for 1 second before checking again
    END LOOP;
END;
/



