USE `pythonlogin_advanced`;
DROP TABLE IF EXISTS `invoice`;
DROP TABLE IF EXISTS `planlist`;
CREATE TABLE IF NOT EXISTS `planlist` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `type` varchar(20) NOT NULL,
  `validity` int(3) NOT NULL,
  `price` int(4) NOT NULL,
  PRIMARY KEY (`id`)
)ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;

INSERT INTO `planlist` (`id`, `type`, `validity`, `price`) VALUES
(1, 'free', 7, 0),
(2, 'individual', 30, 35),
(3, 'individual', 90, 45),
(4, 'individual', 180, 55),
(5, 'individual', 365, 65);

CREATE TABLE IF NOT EXISTS `invoice` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `payment_id` varchar(20),
  `email` varchar(50) NOT NULL,
  `plan_id` int(10) NOT NULL,
  `price` int(10) NOT NULL,
  `status` varchar(10) DEFAULT 'unpaid',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`plan_id`) REFERENCES `planlist`(`id`)
)ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;

DROP TABLE IF EXISTS `membership`;

CREATE TABLE `membership` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `plan_detail` varchar(100) NOT NULL,
  `is_active` BIT(1) NOT NULL,
  `expire_at` DATETIME NOT NULL,
  `start_at` DATETIME NOT NULL,
  `fk_user_account` INT(10) NOT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `membership_ibfk_1` FOREIGN KEY (`fk_user_account`) REFERENCES `accounts`(`id`)
)ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;


INSERT INTO `membership` (`id`, `plan_detail`, `is_active`, `expire_at`, `start_at`, `fk_user_account`) VALUES
(1, 'Admin Unlimited validity plan', b'1', '2050-01-01 00:00:00', '2021-10-30 00:00:00', 1),
(2, 'Admin Unlimited validity plan', b'1', '2050-01-01 00:00:00', '2021-10-30 00:00:00', 2);
