CREATE DATABASE IF NOT EXISTS `pythonlogin_advanced` DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
USE `pythonlogin_advanced`;

DROP TABLE IF EXISTS `membership`;
DROP TABLE IF EXISTS `accounts`;

CREATE TABLE IF NOT EXISTS `accounts` (
`id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `password` varchar(255) NOT NULL,
  `email` varchar(100) NOT NULL,
  `buydate` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `role` enum('Member','Admin') NOT NULL DEFAULT 'Member',
  `activation_code` varchar(255) NOT NULL DEFAULT '',
  `rememberme` varchar(255) NOT NULL DEFAULT '',
  `reset` varchar(255) NOT NULL DEFAULT '',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;


INSERT INTO `accounts` (`id`, `username`, `password`, `email`, `role`, `buydate`, `activation_code`, `rememberme`, `reset`) VALUES
(1, 'admin', 'd0fdf25f69d5d26827ea20729683faae3954357a', 'clarences@aol.com', 'Admin', '2014-07-31 15:42:52', 'activated', '9a7907f87716e70af15e7cd37011c8605843ac25', ''),
(2, 'member', 'd0fdf25f69d5d26827ea20729683faae3954357a', 'luxerin@me.com', 'Member', '2014-07-31 15:42:52', 'activated', 'ot', ''),
(3, 'clarence', 'd0fdf25f69d5d26827ea20729683faae3954357a', 'clarence@luxerin.com', 'Admin', '2014-07-31 15:42:52', 'activated', '', ''),
(4, 'govinda.mekala', 'c50f9059bfc551ec7ee5a5ea758d5dcde94654aa', 'govinda.mekala@eixglobal.com', 'Admin', '2014-07-31 15:42:52', 'activated', '5394f318ac2cee3a866ce4cf52068bbb68898422', ''),
(5, 'Godwin', 'ad1cd583f824d14c34168022ae894d9dd7db642c', 'godwinmuthomim07@gmail.com', 'Admin', '2014-07-31 15:42:52', 'activated', '2178a4da9aa343f149510d9cffe9a1323ba3f1fe', ''),
(8, 'span', '21025e9928d13cb7106ef033b20dc58a292e5f9e', 'span@opentext.com', 'Member', '2014-07-31 15:42:52', 'activated', 'ot', ''),
(19, 'Waqar', '21025e9928d13cb7106ef033b20dc58a292e5f9e', 'waqar@eixglobal.com', 'Member', '2021-09-29 14:53:04', 'activated', 'ot', ''),
(20, 'John', '21025e9928d13cb7106ef033b20dc58a292e5f9e', 'jprice@opentext.com', 'Member', '2021-10-06 18:04:46', 'activated', '', '');

