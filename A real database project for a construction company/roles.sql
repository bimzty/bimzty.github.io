-- ERM the Customer avaliable to Option_List, Decotator Choice, construction_project, lot, Sales_Representative(for that particular customer), Construction_Manager(for that particular customer), and Sale table(only for that particular customer)

DROP ROLE CustomerRole;
CREATE ROLE CustomerRole;

--Assign Permission
GRANT SELECT ON CustomerSaleRecord TO CustomerRole;
GRANT SELECT ON ACCESSIBLE_PROJECTS TO CustomerRole;
GRANT SELECT ON PROJECT_OPTIONS_VIEW TO CustomerRole;
GRANT SELECT ON Option_List TO CustomerRole;
GRANT SELECT ON Decorator_Choice TO CustomerRole;
GRANT SELECT ON Construction_Project TO CustomerRole;
GRANT SELECT ON Lot TO CustomerRole;

--DBA prcess for emplyee
DROP ROLE EmployeeRole;
CREATE ROLE EmployeeRole;

--select, update, delete are granted for most tables
GRANT SELECT, UPDATE, DELETE ON School_District TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Subdivision TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON School TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Elevation TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Lot TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Chosen_Style TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON House_Style TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Room TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Sale TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Bank TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Escrow TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Option_List TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Customer TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Construction_Project TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Task TO EmployeeRole;
GRANT SELECT, UPDATE, DELETE ON Decorator_Choice TO EmployeeRole;
