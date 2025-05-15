SELECT style_name, style_size
FROM House_Style
WHERE style_name = 'Colonial';

INSERT INTO room (room_id, room_name, room_size, floor, room_desc, window_no, ceiling, house_style_style_name)
VALUES (get_next_room_id('Colonial'), 'Attic', 100, 3, 'Attic for storage', 2, 'Coffered', 'Colonial');

SELECT style_name, style_size
FROM House_Style
WHERE style_name = 'Colonial';

ROLLBACK;
