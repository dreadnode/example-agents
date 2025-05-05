<?php

/**
 * Welcome Page Class
 *
 * @copyright   Copyright (c) 2015, Jeffrey Carandang
 * @since       1.0
 */
// Exit if accessed directly
if (!defined('ABSPATH')) exit;
/**
 * WIDGETOPS_Welcome Class
 *
 * A general class for About and Credits page.
 *
 * @since 1.0
 */

if (!class_exists('WIDGETOPS_Welcome')) {
	class WIDGETOPS_Welcome
	{
		public function __construct()
		{
			add_action('admin_enqueue_scripts', array($this, 'enqueue'));
			add_action('admin_menu', array($this, 'screen_page'));
			// add_action( 'activated_plugin', array($this, 'redirect' ) );
			add_action('admin_head', array($this, 'remove_menu'));
		}

		function enqueue()
		{
			if (!isset($_GET['page']) || 'widget-opts-getting-started' != $_GET['page'])
				return;

			wp_enqueue_style('extended-widget-opts-welcome', plugins_url('../assets/css/welcome.css', dirname(__FILE__)), array(), WIDGETOPTS_VERSION);
		}

		function screen_page()
		{
			add_dashboard_page(
				__('Getting started with Widget Options', 'widget-options'),
				__('Getting started with Widget Options', 'widget-options'),
				apply_filters('widgetopts_welcome_cap', 'manage_options'),
				'widget-opts-getting-started',
				array($this, 'welcome_content')
			);
		}

		function welcome_head()
		{
			$selected = isset($_GET['page']) ? $_GET['page'] : 'widget-opts-getting-started';
?>
			<h1><?php _e('Welcome to Widget Options', 'widget-options'); ?></h1>
			<div class="about-text">
				<?php _e('Congratulations! You\'ve just unlocked features on managing your widgets better.', 'widget-options'); ?>
			</div>
			<div class="widgetopts-badge">
				<span class="widgetopts-mascot"></span>
				<span class="version"><?php _e('Version', 'widget-options'); ?> <?php echo WIDGETOPTS_VERSION; ?></span>
			</div>
			<h2 class="nav-tab-wrapper">
				<a class="nav-tab <?php echo $selected == 'widget-opts-getting-started' ? 'nav-tab-active' : ''; ?>" href="<?php echo esc_url(admin_url(add_query_arg(array('page' => 'widget-opts-getting-started'), 'index.php'))); ?>">
					<?php _e('Getting Started', 'widget-options'); ?>
				</a>
			</h2>
		<?php
		}

		function welcome_content()
		{ ?>
			<div class="wrap about-wrap widgetopts-about-wrap">
				<?php $this->welcome_head(); ?>
				<p class="about-description">
					<?php _e('Use the tips and instructions below to get started, then you will be up and running in no time. ', 'widget-options'); ?>
				</p>

				<div class="feature-section two-col">
					<div class="col">
						<h3><?php _e('Take full control over your widgets!', 'widget-options'); ?></h3>
						<p><?php printf(__('Widget Options is your all-in-on plugin to take over your widgets like it was built as WordPress core functionality. You can follow the tutorial on the right to see how the plugin works but in reality it\'s so easy and integrated elegantly on each widgets. <a href="%s" target="_blank">Expand your widgets here to view Widget Options</a>.', 'widget-options'), esc_url(admin_url('widgets.php'))); ?>
					</div>
					<div class="col">
						<div class="feature-video">
							<iframe width="495" height="278" src="https://player.vimeo.com/video/190057410" frameborder="0" allowfullscreen></iframe>
						</div>
					</div>
				</div>

				<div class="feature-section two-col">
					<h3><?php _e('Extend Plugin Features', 'widget-options'); ?></h3>
					<p><?php _e('Unlock more features by upgrading to Widget Options Extended to get the full plugin functionalities! We have tons of helpful features that will let you fully manage your widgets and extend functionalities.', 'widget-options'); ?></p>
					<p><a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE) . 'features/'); ?>" target="_blank" class="widgetopts-features-button button button-primary"><?php _e('See all Features', 'widget-options'); ?></a></p>
				</div>
			</div>
<?php }

		function redirect($plugin)
		{
			if ($plugin == 'widget-options/plugin.php' && !isset($_GET['activate-multi'])) {
				wp_redirect(admin_url('index.php?page=widget-opts-getting-started'));
				die();
			}
		}
		function remove_menu()
		{
			remove_submenu_page('index.php', 'widget-opts-getting-started');
		}
	}
	new WIDGETOPS_Welcome();
}

?>