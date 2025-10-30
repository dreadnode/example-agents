<?php
/**
 * Links Settings Module
 * Settings > Widget Options :: Link Widget
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Custom Links Options
 *
 * @since 3.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_links' ) ):
	function widgetopts_settings_links(){ ?>
		<li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="links">
			<div class="widgetopts-module-card-content">
				<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'features/link-wordpress-widgets/');?>" target="_blank" class="widgetopts-pro-upsell"></a>
				<h2><?php _e( 'Link Widget', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label"><span class="dashicons dashicons-lock"></span></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Add custom links to any widgets to redirect users on click action.', 'widget-options' );?>
				</p>
			</div>
		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_links', 70 );
endif;
?>
