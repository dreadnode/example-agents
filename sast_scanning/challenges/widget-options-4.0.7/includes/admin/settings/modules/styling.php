<?php
/**
 * Styling Settings Module
 * Settings > Widget Options :: Custom Styling
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Custom Styling Options
 *
 * @since 3.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_styling' ) ):
	function widgetopts_settings_styling(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="styling">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/wordpress-widgets-styling/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'Custom Styling', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Set custom widget colors and styling to make your widget stand-out more.', 'widget-options' );?>
				</p>
			</div>
		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_styling', 120 );
endif;
?>
