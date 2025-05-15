-- ========================== Scheduled Jobs: Options Availabile==========================
-- Drop the scheduled job: YEARLY_INACTIVE_CUSTOMER_CLEANUP
BEGIN
    DBMS_SCHEDULER.DROP_JOB('YEARLY_INACTIVE_CUSTOMER_CLEANUP');
END;
/

-- Procedure deletes
CREATE OR REPLACE PROCEDURE sp_delete_inactive_customers AS
  v_deleted_count NUMBER;
BEGIN
  -- Delete the customers who have not made a sale
  DELETE FROM customer c
  WHERE NOT EXISTS (
      SELECT 1
      FROM sale s
      WHERE s.customer_customer_id = c.customer_id
  );

  -- Get the number of rows deleted
  v_deleted_count := SQL%ROWCOUNT;
  dbms_output.put_line('Removed ' || v_deleted_count || ' customers who have not purchased a lot.');
END;
/

-- Create scheduled job
BEGIN
  DBMS_SCHEDULER.CREATE_JOB (
    job_name        => 'YEARLY_INACTIVE_CUSTOMER_CLEANUP',
    job_type        => 'STORED_PROCEDURE',
    job_action      => 'sp_delete_inactive_customers',
    start_date      => SYSDATE,
    repeat_interval => 'FREQ=YEARLY',
    enabled         => TRUE,
    comments        => 'Job to delete inactive customers annually'
  );
END;
/

-- -- ========================== Scheduled Jobs: Options Availabile==========================
