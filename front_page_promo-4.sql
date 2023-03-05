-- phpMyAdmin SQL Dump
-- version 4.9.5deb2
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Dec 22, 2021 at 03:01 AM
-- Server version: 8.0.27-0ubuntu0.20.04.1
-- PHP Version: 7.4.26

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `pythonlogin_advanced`
--

-- --------------------------------------------------------

--
-- Table structure for table `front_page_promo`
--

CREATE TABLE `front_page_promo` (
  `id` int NOT NULL,
  `promo_email` varchar(100) NOT NULL,
  `promo_date` date DEFAULT NULL,
  `times_used` int NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;

--
-- Dumping data for table `front_page_promo`
--

INSERT INTO `front_page_promo` (`id`, `promo_email`, `promo_date`, `times_used`) VALUES
(1, 'clarences@aol.com', '2021-12-15', 5),
(2, 'godwinmuthomim07@gmail.com', '2021-12-15', 6),
(3, 'clarence@luxerin.com', '2021-12-15', 6),
(4, 'luxerin@me.com', '2021-12-20', 5),
(6, 'omokabet@gmail.com', '2021-12-07', 4),
(7, 'godwinmuthomim7@gmail.com', '2021-12-15', 5),
(8, 'francisco.mwendak@gmail.com', '2021-11-25', 1);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `front_page_promo`
--
ALTER TABLE `front_page_promo`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `front_page_promo`
--
ALTER TABLE `front_page_promo`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
