<?php

/**
 * Admin Options Page
 * Settings > Widget Options
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       4.1
 */
// Exit if accessed directly
if (!defined('ABSPATH')) exit;

/**
 * Creates the admin submenu pages under the Settings menu and assigns their
 *
 * @since 1.0
 * @return void
 */
if (!function_exists('widgetopts_add_options_link')) :
	function widgetopts_add_options_link()
	{
		add_options_page(
			__('Widget Options', 'widget-options'),
			__('Widget Options', 'widget-options'),
			'manage_options',
			'widgetopts_plugin_settings',
			'widgetopts_options_page'
		);
	}
	add_action('admin_menu', 'widgetopts_add_options_link', 10);
endif;

/**
 * Options Page
 *
 * Renders the options page contents.
 *
 * @since 1.0
 * @return void
 */
if (!function_exists('widgetopts_options_page')) :
	function widgetopts_options_page()
	{
		$view = 'grid'; //define so that we can add more views later on
		$upgrade_url = apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE) . '?utm_source=wordpressadmin&utm_medium=widget&utm_campaign=widgetoptsproupgrade');
?>
		<div class="wrap">
			<h1>
				<?php _e('Widget Options', 'widget-options'); ?>
				<a href="<?php echo esc_url(apply_filters('widget_options_support_url', 'https://wordpress.org/support/plugin/widget-options/')); ?>" target="_blank" class="page-title-action"><?php _e('Support', 'widget-options'); ?></a>
				<a href="<?php echo esc_url(apply_filters('widget_options_upgrade_url', $upgrade_url)); ?>" target="_blank" class="page-title-action"><?php _e('Upgrade', 'widget-options'); ?></a>
			</h1>

			<div id="widgetopts-settings-messages-container"></div>
			<div class="widgetopts-settings-desc">
				<?php _e('Enable or disable any widget options tabs using this option. Some features has settings configuration that you can take advantage of to get the most out of Widget Options Extended on fully managing your widgets.', 'widget-options'); ?>
			</div>
			<div class="widgetopts-badge widgetopts-badge-settings">
				<span class="widgetopts-mascot"></span>
			</div>

			<!-- <div class="widgetopts-get-latest-updates">
				<h2><?php _e('Get the latest updates and features from Widget Options!', 'widget-options'); ?></h2>
				<h2><?php _e('Email', 'widget-options'); ?>:</h2>
				<input type="email" name="widgetopts_update_and_feaures_email" />
				<button id="widgetoptsSubmitUpdateFeaturesEmail" class="button button-secondary">Submit</button>
			</div> -->

			<div id="poststuff" class="widgetopts-poststuff">
				<div id="post-body" class="metabox-holder columns-2 hide-if-no-js">
					<div id="postbox-container-2" class="postbox-container">

						<div class="widgetopts-module-cards-container <?php echo $view; ?> hide-if-no-js">
							<form enctype="multipart/form-data" method="post" action="/wp-admin/admin.php?page=widgetopts_plugin_settings" id="widgetopts-module-settings-form">
								<ul class="widgetopts-module-cards">
									<?php echo do_action('widgetopts_module_cards'); ?>
								</ul>
							</form>
						</div>
						<div class="widgetopts-modal-background"></div>
					</div>

					<div id="postbox-container-1" class="postbox-container">
						<?php echo do_action('widgetopts_module_sidebar'); ?>
					</div>

				</div>
			</div>
		</div>
	<?php
	}
endif;

/**
 * Modal Wrapper
 *
 * Create callable modal wrappers to avoid writing same code again
 *
 * @since 4.0
 * @return void
 */
if (!function_exists('widgetopts_modal_start')) :
	function widgetopts_modal_start($option = null)
	{ ?>
		<div class="widgetopts-module-settings-container">
			<div class="widgetopts-modal-navigation">
				<button class="dashicons widgetopts-close-modal"></button>
			</div>
			<div class="widgetopts-module-settings-content-container">
				<div class="widgetopts-module-settings-content">
				<?php }
		endif;

		if (!function_exists('widgetopts_modal_end')) :
			function widgetopts_modal_end($option = null)
			{ ?>
				</div>
			</div>
			<div class="widgetopts-list-content-footer hide-if-no-js">
				<button class="button button-primary align-left widgetopts-module-settings-save"><?php _e('Save Settings', 'widget-options'); ?></button>
				<button class="button button-secondary align-left widgetopts-module-settings-cancel"><?php _e('Cancel', 'widget-options'); ?></button>
			</div>
			<div class="widgetopts-modal-content-footer">
				<?php if ($option == 'activate') { ?>
					<button class="button button-secondary align-right widgetopts-toggle-activation"><?php _e('Disable', 'widget-options'); ?></button>
				<?php } else { ?>
					<button class="button button-primary align-right widgetopts-toggle-activation"><?php _e('Enable', 'widget-options'); ?></button>
				<?php } ?>
				<button class="button button-primary align-left widgetopts-module-settings-save"><?php _e('Save Settings', 'widget-options'); ?></button>
			</div>
		</div>
<?php }
		endif; ?>